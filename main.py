#!/usr/bin/env python3
"""
=============================================================================
 DISASTER RESCUE DRONE -- Raspberry Pi companion computer
 Single script, single process tree, single entry point.
=============================================================================

    python3 main.py --sitl                    # ArduPilot SITL, no hardware
    python3 main.py --config drone.yaml
    python3 main.py --goal 12 8 --no-display

CONCURRENCY MODEL
-----------------
  MAIN PROCESS (threads -- all I/O-bound, so the GIL costs nothing)
    lidar_rx      YDLidar SDK; doProcessSimple releases the GIL in C
    mavlink_rx    blocks on socket read
    vision_tx     VISION_POSITION_ESTIMATE to the FC on a timer
    mav_proxy     forwards FC <-> GCS so Mission Planner still works
    telemetry     ZMQ PUB; releases the GIL on send
    watchdog      sleeps almost always; the supervisor

  WORLD PROCESS (exactly one child -- CPU-bound, owns a core)
    ICP scan matching, occupancy grid integration, A* + coverage planning

WHY SLAM IS A PROCESS AND NOT A THREAD
  ICP is 10-25 ms of KDTree queries and Python loop overhead that HOLD the
  GIL. At 7 Hz that is a ~17% duty cycle of GIL-hogging bursts, which would
  inject jitter into mavlink_rx -- the thread feeding position to the flight
  controller. A process decouples them entirely.

WHY SLAM AND THE PLANNER SHARE ONE PROCESS
  Both need the occupancy grid. Split apart, a 240x240 float32 grid (230 KB)
  crosses a process boundary on every planner tick. Together they share it as
  an ordinary numpy array for free. Planning is event-driven, not continuous,
  so it fits between scan integrations.

IPC: shared_memory + seqlock for the grid, mp.Array for pose, bounded
mp.Queue (drop-oldest) for scans and commands. NOT multiprocessing.Manager --
every Manager access is a socket round trip plus pickling, ~50-100 us, and it
spawns a whole extra process to schedule.

-----------------------------------------------------------------------------
ONE-TIME OS SETUP -- NOT managed by this script
-----------------------------------------------------------------------------
The A8 mini serves RTSP itself. This Pi ROUTES that stream; it never decodes
or re-encodes it (that would cost 1-2 cores to convert H.265 into H.265).
Run once on the Pi:

    sudo sysctl -w net.ipv4.ip_forward=1
    sudo iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
    sudo iptables -A FORWARD -i wlan0 -o eth0 -j ACCEPT
    sudo iptables -A FORWARD -i eth0 -o wlan0 -m state --state RELATED,ESTABLISHED -j ACCEPT
    sudo apt install -y iptables-persistent && sudo netfilter-persistent save
    echo "net.ipv4.ip_forward=1" | sudo tee -a /etc/sysctl.conf

On the GCS laptop, route the camera subnet via the Pi, then open
rtsp://192.168.144.25:8554/main.264 directly. Video never enters this script.

-----------------------------------------------------------------------------
DEPENDENCIES
    sudo apt install -y python3-numpy python3-scipy python3-yaml
    pip install pymavlink pyzmq msgpack --break-system-packages
    # ydlidar: build YDLidar-SDK, then pip install . from its root

scipy is MANDATORY. Its cKDTree makes ICP 11x faster than a numpy brute-force
search (4 ms vs 46 ms per match). Without it, scan matching alone would eat
115 ms of the 143 ms available per scan on a Pi 4B.

-----------------------------------------------------------------------------
REQUIRED ArduPilot PARAMETERS (indoor / GPS-denied)
    VISO_TYPE        1     MAVLink vision odometry
    VISO_DELAY_MS    60    MEASURE THIS. Reported at shutdown as p50 latency.
    EK3_SRC1_POSXY   6     ExternalNav  <-- makes this script flight-critical
    EK3_SRC1_VELXY   6
    EK3_SRC1_POSZ    1     baro; the 2D lidar knows nothing about altitude
    EK3_SRC1_YAW     1     compass. Do NOT use vision yaw -- see note below.
    PRX1_TYPE        2     MAVLink proximity, from OBSTACLE_DISTANCE
    AVOID_ENABLE     7     FC-side avoidance, independent of this script

Vision yaw is deliberately not sent. ICP yaw error is ~0.5 deg per scan and
integrates without loop closure; feeding it to EKF3 slowly rotates your map.
Position only.
=============================================================================
"""

from __future__ import annotations

import argparse
import ctypes
import logging
import math
import multiprocessing as mp
import multiprocessing.shared_memory as shm
import os
import queue
import signal
import socket
import sys
import threading
import time
import zlib
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

try:
    from scipy.spatial import cKDTree
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

log = logging.getLogger("drone")

# ===========================================================================
# CONFIG
# ===========================================================================

CONFIG = {
    "mavlink": {"device": "/dev/serial0", "baud": 921600,
                "proxy_out": "udpout:192.168.1.255:14550", "proxy": True},
    "lidar": {"model": "X2L", "port": "/dev/ydlidar", "enabled": True},
    "slam": {
        "icp_iters": 30,          # 12 under-converges: 12 deg came back as 7.6
        "icp_tol": 1e-4,
        "icp_trim": 0.75,         # drop worst 25% of correspondences
        "subsample": 1,           # raise to 2 if CPU-bound; halves cost
        "max_match_dist": 1.0,    # correspondences beyond this are outliers
        "min_points": 40,
        "max_fitness": 0.25,      # mean residual above this -> reject match
        "keyframe_dist": 0.30,    # metres before the reference scan advances
        "keyframe_yaw": 0.20,     # radians
    },
    "map": {"size_m": 40.0, "resolution": 0.10,   # indoor: finer than outdoor
            "max_tilt_deg": 10.0, "max_range": 8.0, "min_range": 0.12,
            "ground_margin": 0.8},
    "vehicle": {"radius": 0.45, "safety_margin": 0.45,
                "decel": 1.5, "reaction": 0.35},
    "flight": {"takeoff_alt": 1.5, "cruise_speed": 0.8, "wp_tolerance": 0.4,
               "geofence_radius": 40.0, "max_alt": 4.0, "min_battery_pct": 25.0},
    "vision": {"send_hz": 12.0,   # EKF3 wants >=10; lidar is 7, so extrapolate
               "enabled": True, "reset_counter_on_jump": True},
    "limits": {"max_pose_age": 0.5, "max_scan_age": 1.5,
               "max_slam_age": 1.0, "replan_attempts": 4},
    "telemetry": {"enabled": True, "bind": "tcp://0.0.0.0:5556",
                  "map_hz": 2.0, "scan_hz": 5.0, "state_hz": 10.0,
                  "obstacle_hz": 8.0},
    "display": {"enabled": False, "rate_hz": 4.0},
}


def deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = deep_merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


class FlightState(Enum):
    BOOT = auto(); PREFLIGHT = auto(); ARMING = auto(); TAKEOFF = auto()
    NAVIGATE = auto(); HOLD = auto(); LANDING = auto(); LANDED = auto(); ABORT = auto()


class Abort(Enum):
    NONE = auto(); OPERATOR = auto(); POSE_TIMEOUT = auto(); LIDAR_TIMEOUT = auto()
    SLAM_LOST = auto(); GEOFENCE = auto(); LOW_BATTERY = auto()
    NO_PATH = auto(); FC_DISARMED = auto(); WORLD_DIED = auto()


# ===========================================================================
# FRAME CONVENTIONS
#   World: ENU metres, origin at takeoff. x=East y=North z=Up.
#   Yaw:   radians, 0 = East, CCW positive (math convention).
#   MAVLink is NED with compass yaw. Conversion happens ONLY at that boundary.
#   Time:  time.monotonic() everywhere. Never wall clock -- an NTP step
#          mid-flight would corrupt pose interpolation.
# ===========================================================================

def wrap_pi(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def enu_to_ned(x, y, z):
    return (y, x, -z)


def yaw_ned_to_enu(y):
    return wrap_pi(math.pi / 2 - y)


@dataclass
class Pose:
    t: float = 0.0
    x: float = 0.0; y: float = 0.0; z: float = 0.0
    vx: float = 0.0; vy: float = 0.0; vz: float = 0.0
    roll: float = 0.0; pitch: float = 0.0; yaw: float = 0.0
    rel_alt: float = 0.0
    valid: bool = False

    def tilt(self) -> float:
        return math.hypot(self.roll, self.pitch)


# ===========================================================================
# SHARED STATE  --  shared_memory + seqlock, not a Manager
# ===========================================================================

POSE_N = 12   # t,x,y,z,vx,vy,vz,roll,pitch,yaw,fitness,valid


class SharedState:
    """
    Grid lives in shared_memory; pose in an mp.Array. Both use a seqlock:
    the writer bumps a counter to odd, writes, bumps to even. A reader that
    sees odd, or a changed counter across its copy, retries. Lock-free, and
    a slow reader can never stall a sensor loop.
    """

    def __init__(self, grid_n: int):
        self.grid_n = grid_n
        nbytes = grid_n * grid_n * 4
        self._shm = shm.SharedMemory(create=True, size=nbytes)
        self.grid = np.ndarray((grid_n, grid_n), dtype=np.float32, buffer=self._shm.buf)
        self.grid[...] = 0.0
        self.grid_name = self._shm.name

        self._gseq = mp.Value(ctypes.c_uint64, 0, lock=False)
        self._gmeta = mp.Array(ctypes.c_double, 4, lock=False)   # t, res, ox, oy
        self._pose = mp.Array(ctypes.c_double, POSE_N, lock=False)
        self._pseq = mp.Value(ctypes.c_uint64, 0, lock=False)
        self.slam_stamp = mp.Value(ctypes.c_double, 0.0, lock=False)
        self.world_alive = mp.Value(ctypes.c_double, 0.0, lock=False)

    # ---- grid ----
    def write_grid(self, g: np.ndarray, res: float, ox: float, oy: float) -> None:
        self._gseq.value += 1
        self.grid[...] = g
        self._gmeta[0] = time.monotonic(); self._gmeta[1] = res
        self._gmeta[2] = ox; self._gmeta[3] = oy
        self._gseq.value += 1

    def read_grid(self, retries: int = 6):
        for _ in range(retries):
            s0 = self._gseq.value
            if s0 & 1:
                continue
            g = self.grid.copy()
            meta = (self._gmeta[0], self._gmeta[1], self._gmeta[2], self._gmeta[3])
            if self._gseq.value == s0:
                return g, meta
        return None, (0.0, 0.0, 0.0, 0.0)

    def grid_age(self) -> float:
        t = self._gmeta[0]
        return 999.0 if t == 0.0 else time.monotonic() - t

    # ---- pose ----
    def write_pose(self, p: Pose, fitness: float = 0.0) -> None:
        self._pseq.value += 1
        a = self._pose
        a[0] = p.t; a[1] = p.x; a[2] = p.y; a[3] = p.z
        a[4] = p.vx; a[5] = p.vy; a[6] = p.vz
        a[7] = p.roll; a[8] = p.pitch; a[9] = p.yaw
        a[10] = fitness; a[11] = 1.0 if p.valid else 0.0
        self._pseq.value += 1

    def read_pose(self, retries: int = 6):
        for _ in range(retries):
            s0 = self._pseq.value
            if s0 & 1:
                continue
            a = list(self._pose)
            if self._pseq.value == s0:
                p = Pose(t=a[0], x=a[1], y=a[2], z=a[3], vx=a[4], vy=a[5], vz=a[6],
                         roll=a[7], pitch=a[8], yaw=a[9], valid=a[11] > 0.5)
                return p, a[10]
        return None, 0.0

    def close(self) -> None:
        try:
            self._shm.close(); self._shm.unlink()
        except (FileNotFoundError, BufferError):
            pass


# ===========================================================================
# SCAN MATCHER  --  the SLAM front end
# ===========================================================================

class ScanMatcher:
    """
    Point-to-point ICP with trimmed correspondences.

    Benchmarked: 30 iterations with a cKDTree costs ~10 ms on x86, roughly
    25 ms on a Pi 4B, against a 143 ms budget at 7 Hz. A 12-iteration cap was
    tried and rejected -- it under-converges badly (a true 12 deg rotation came
    back as 7.6 deg), and systematic under-estimation of rotation becomes
    heading drift.

    Seeded from a constant-velocity model, which roughly halves the iterations
    needed. Deliberately NOT seeded from the FC's fused pose: EKF3 is already
    consuming our output, so seeding from it would close a feedback loop where
    our own errors reinforce themselves.
    """

    def __init__(self, cfg: dict):
        c = cfg["slam"]
        self.iters = c["icp_iters"]; self.tol = c["icp_tol"]; self.trim = c["icp_trim"]
        self.sub = max(1, c["subsample"]); self.max_dist = c["max_match_dist"]
        self.min_points = c["min_points"]; self.max_fitness = c["max_fitness"]
        self.kf_dist = c["keyframe_dist"]; self.kf_yaw = c["keyframe_yaw"]

        self.ref: np.ndarray | None = None
        self.ref_tree = None
        self.ref_pose = (0.0, 0.0, 0.0)
        self.pose = np.array([0.0, 0.0, 0.0])       # x, y, yaw -- world ENU
        self.vel = np.array([0.0, 0.0, 0.0])        # for the seed
        self.last_t: float | None = None
        self.matches = 0; self.rejects = 0

    @staticmethod
    def polar_to_xy(ang, rng):
        return np.stack([rng * np.cos(ang), rng * np.sin(ang)], axis=1)

    def _nn(self, tree, ref, Q):
        if tree is not None:
            d, i = tree.query(Q)
            return i, d
        dd = ((Q[:, None, :] - ref[None, :, :]) ** 2).sum(-1)
        i = np.argmin(dd, axis=1)
        return i, np.sqrt(dd[np.arange(len(Q)), i])

    @staticmethod
    def _kabsch(Q, P):
        """Closed-form optimal 2D transform taking Q onto P. No SVD needed."""
        qm, pm = Q.mean(0), P.mean(0)
        Qc, Pc = Q - qm, P - pm
        num = float((Qc[:, 0] * Pc[:, 1] - Qc[:, 1] * Pc[:, 0]).sum())
        den = float((Qc[:, 0] * Pc[:, 0] + Qc[:, 1] * Pc[:, 1]).sum())
        th = math.atan2(num, den)
        c, s = math.cos(th), math.sin(th)
        R = np.array([[c, -s], [s, c]])
        return th, pm - R @ qm

    def _icp(self, P, Q, tree, init):
        th = init[2]
        t = np.array([init[0], init[1]], float)
        fitness = 9.9
        used = 0
        for used in range(1, self.iters + 1):
            c, s = math.cos(th), math.sin(th)
            R = np.array([[c, -s], [s, c]])
            Qt = Q @ R.T + t
            idx, dist = self._nn(tree, P, Qt)
            ok = dist < self.max_dist
            if ok.sum() < self.min_points:
                return None
            cut = np.quantile(dist[ok], self.trim)
            keep = ok & (dist <= cut)
            if keep.sum() < self.min_points:
                keep = ok
            dth, dt = self._kabsch(Qt[keep], P[idx[keep]])
            c2, s2 = math.cos(dth), math.sin(dth)
            t = np.array([[c2, -s2], [s2, c2]]) @ t + dt
            th = wrap_pi(th + dth)
            fitness = float(dist[keep].mean())
            if abs(dth) < self.tol and float(np.hypot(*dt)) < self.tol:
                break
        return t[0], t[1], th, fitness, used

    def update(self, t_scan: float, angles: np.ndarray, ranges: np.ndarray,
               min_range: float, max_range: float):
        """Returns (x, y, yaw, fitness) in world ENU, or None if unusable."""
        valid = np.isfinite(ranges) & (ranges > min_range) & (ranges < max_range)
        if valid.sum() < self.min_points:
            self.rejects += 1
            return None
        pts = self.polar_to_xy(angles[valid].astype(np.float64),
                               ranges[valid].astype(np.float64))[::self.sub]
        if len(pts) < self.min_points:
            self.rejects += 1
            return None

        if self.ref is None:
            self.ref = pts
            self.ref_tree = cKDTree(pts) if HAVE_SCIPY else None
            self.ref_pose = tuple(self.pose)
            self.last_t = t_scan
            self.matches += 1
            return (self.pose[0], self.pose[1], self.pose[2], 0.0)

        # NOTE: `is None`, never truthiness. A timestamp of exactly 0.0 is a
        # valid time, and `self.last_t or t_scan` would treat it as unset --
        # collapsing dt to ~0, which sends the velocity estimate to absurd
        # values and throws the ICP seed metres away from truth. Every
        # subsequent scan then fails the outlier test and matching dies.
        prev_t = self.last_t if self.last_t is not None else t_scan
        dt = min(1.0, max(0.02, t_scan - prev_t))
        # constant-velocity seed, expressed in the reference scan's frame
        seed_world = self.pose + self.vel * dt
        dx_w = seed_world[0] - self.ref_pose[0]
        dy_w = seed_world[1] - self.ref_pose[1]
        rth = self.ref_pose[2]
        c, s = math.cos(-rth), math.sin(-rth)
        init = (c * dx_w - s * dy_w, s * dx_w + c * dy_w,
                wrap_pi(seed_world[2] - rth))

        res = self._icp(self.ref, pts, self.ref_tree, init)
        if res is None:
            self.rejects += 1
            return None
        lx, ly, lth, fitness, _ = res
        if fitness > self.max_fitness:
            self.rejects += 1
            return None

        # local (reference-frame) result back into world ENU
        c, s = math.cos(rth), math.sin(rth)
        nx = self.ref_pose[0] + c * lx - s * ly
        ny = self.ref_pose[1] + s * lx + c * ly
        nth = wrap_pi(rth + lth)

        new_pose = np.array([nx, ny, nth])
        # Clamp the seed velocity to what an indoor multirotor can physically
        # do. This is the safety net: even with a correct dt, one bad match
        # could otherwise produce a wild velocity that poisons the seed for
        # every scan after it.
        raw_v = np.array([(nx - self.pose[0]) / dt, (ny - self.pose[1]) / dt,
                          wrap_pi(nth - self.pose[2]) / dt])
        self.vel = np.clip(raw_v, [-5.0, -5.0, -3.0], [5.0, 5.0, 3.0])
        self.pose = new_pose
        self.last_t = t_scan
        self.matches += 1

        # advance the reference once we have moved enough -- matching against
        # a keyframe rather than every scan keeps per-scan error from
        # accumulating as fast as pure frame-to-frame would
        if (math.hypot(nx - self.ref_pose[0], ny - self.ref_pose[1]) > self.kf_dist
                or abs(wrap_pi(nth - self.ref_pose[2])) > self.kf_yaw):
            self.ref = pts
            self.ref_tree = cKDTree(pts) if HAVE_SCIPY else None
            self.ref_pose = (nx, ny, nth)

        return (nx, ny, nth, fitness)


# ===========================================================================
# OCCUPANCY GRID  --  2D birds-eye, log-odds, vehicle-centred rolling window
# ===========================================================================

L_OCC, L_FREE, L_MIN, L_MAX, L_DECAY = 0.85, -0.40, -4.0, 4.0, 0.02


class OccupancyGrid2D:
    """
    A 2D lidar assumes its scan plane stays level; a multirotor tilts to move.
    Two consequences, both handled here:

      RANGE STRETCH  a wall at true distance d reads d/cos(tilt). Corrected.
      GROUND STRIKE  tilted by theta at altitude h, the beam hits the floor at
                     h/sin(theta). At 1.5 m indoors and 8 deg that is 10.8 m --
                     within range, painting a phantom arc across the flight
                     path. Returns are tested against the floor intersection
                     and discarded.

    Insertion is endpoint-only with global decay, not raycast clearing.
    Raycasting ~360 beams through a 400x400 grid is ~10^5 cell updates per
    scan; far too slow in Python at 7 Hz. Decay recovers most of the benefit.
    """

    def __init__(self, cfg: dict):
        m = cfg["map"]
        self.res = m["resolution"]
        self.n = int(round(m["size_m"] / self.res))
        self.max_tilt = math.radians(m["max_tilt_deg"])
        self.min_range, self.max_range = m["min_range"], m["max_range"]
        self.ground_margin = m["ground_margin"]
        self.log_odds = np.zeros((self.n, self.n), dtype=np.float32)
        self.origin = np.array([0.0, 0.0])
        self.updates = 0; self.rej_tilt = 0; self.rej_ground = 0
        self.last_update = 0.0

    def world_to_cell(self, x, y):
        cx = (np.asarray(x) - self.origin[0]) / self.res + self.n / 2.0
        cy = (np.asarray(y) - self.origin[1]) / self.res + self.n / 2.0
        return np.floor(cx).astype(np.int32), np.floor(cy).astype(np.int32)

    def cell_to_world(self, col, row):
        x = (np.asarray(col) - self.n / 2.0 + 0.5) * self.res + self.origin[0]
        y = (np.asarray(row) - self.n / 2.0 + 0.5) * self.res + self.origin[1]
        return x, y

    def recenter(self, x: float, y: float) -> None:
        dxc = int(round((x - self.origin[0]) / self.res))
        dyc = int(round((y - self.origin[1]) / self.res))
        if abs(dxc) < self.n // 8 and abs(dyc) < self.n // 8:
            return
        self.log_odds = np.roll(self.log_odds, (-dyc, -dxc), axis=(0, 1))
        if dxc > 0:   self.log_odds[:, -dxc:] = 0.0
        elif dxc < 0: self.log_odds[:, :-dxc] = 0.0
        if dyc > 0:   self.log_odds[-dyc:, :] = 0.0
        elif dyc < 0: self.log_odds[:-dyc, :] = 0.0
        self.origin += np.array([dxc * self.res, dyc * self.res])

    def integrate(self, angles, ranges, x, y, yaw, tilt, agl) -> bool:
        if tilt > self.max_tilt:
            self.rej_tilt += 1
            return False
        valid = np.isfinite(ranges) & (ranges > self.min_range) & (ranges < self.max_range)
        if valid.sum() < 10:
            return False
        a = angles[valid].astype(np.float64)
        r = ranges[valid].astype(np.float64) * math.cos(tilt)   # de-stretch

        if agl > 0.05 and tilt > math.radians(0.5):             # floor rejection
            down = np.abs(np.sin(a - yaw) * math.sin(tilt)) + 1e-9
            keep = r < (agl / down - self.ground_margin)
            self.rej_ground += int((~keep).sum())
            a, r = a[keep], r[keep]
            if a.size == 0:
                return False

        wa = a + yaw
        px, py = x + r * np.cos(wa), y + r * np.sin(wa)
        self.recenter(x, y)
        col, row = self.world_to_cell(px, py)
        inside = (col >= 0) & (col < self.n) & (row >= 0) & (row < self.n)
        col, row = col[inside], row[inside]
        if col.size == 0:
            return False

        np.multiply(self.log_odds, 1.0 - L_DECAY, out=self.log_odds)
        np.add.at(self.log_odds, (row, col), L_OCC)
        vc, vr = self.world_to_cell(x, y)
        if 0 <= vc < self.n and 0 <= vr < self.n:
            k = max(1, int(0.5 / self.res))
            self.log_odds[max(0, vr - k):vr + k + 1, max(0, vc - k):vc + k + 1] += L_FREE
        np.clip(self.log_odds, L_MIN, L_MAX, out=self.log_odds)
        self.updates += 1
        self.last_update = time.monotonic()
        return True

    def probability(self):
        return 1.0 - 1.0 / (1.0 + np.exp(self.log_odds))

    def obstacles(self, thr=0.65):
        return self.probability() > thr

    def inflated(self, radius_m, thr=0.65):
        occ = self.obstacles(thr)
        k = int(math.ceil(radius_m / self.res))
        if k <= 0:
            return occ
        out = occ.copy()
        for axis in (0, 1):
            acc = out.copy()
            for s in range(1, k + 1):
                acc |= np.roll(out, s, axis=axis); acc |= np.roll(out, -s, axis=axis)
            out = acc
        return out


# ===========================================================================
# PLANNER  --  coarse A* + a per-tick corridor check
# ===========================================================================

SQRT2 = math.sqrt(2.0)
NBRS = [(-1,0,1.0),(1,0,1.0),(0,-1,1.0),(0,1,1.0),
        (-1,-1,SQRT2),(-1,1,SQRT2),(1,-1,SQRT2),(1,1,SQRT2)]


def astar(blocked, start, goal, max_expand=60000):
    import heapq
    nr, nc = blocked.shape
    sr, sc = start; gr, gc = goal
    if not (0 <= sr < nr and 0 <= sc < nc and 0 <= gr < nr and 0 <= gc < nc):
        return None
    if blocked[gr, gc]:
        free = np.argwhere(~blocked)
        if free.size == 0:
            return None
        d = np.abs(free[:, 0] - gr) + np.abs(free[:, 1] - gc)
        i = int(np.argmin(d))
        if d[i] > 40:
            return None
        gr, gc = int(free[i, 0]), int(free[i, 1])

    def h(r, c):
        dr, dc = abs(r - gr), abs(c - gc)
        return (dr + dc) + (SQRT2 - 2.0) * min(dr, dc)

    openh = [(h(sr, sc), 0.0, (sr, sc))]
    came = {}; g_of = {(sr, sc): 0.0}; closed = set(); n_exp = 0
    while openh:
        _, g, cur = heapq.heappop(openh)
        if cur in closed:
            continue
        closed.add(cur)
        if cur == (gr, gc):
            path = [cur]
            while cur in came:
                cur = came[cur]; path.append(cur)
            return path[::-1]
        n_exp += 1
        if n_exp > max_expand:
            return None
        r, c = cur
        for dr, dc, step in NBRS:
            r2, c2 = r + dr, c + dc
            if not (0 <= r2 < nr and 0 <= c2 < nc) or blocked[r2, c2]:
                continue
            if dr and dc and (blocked[r, c2] or blocked[r2, c]):
                continue
            ng = g + step
            if ng < g_of.get((r2, c2), math.inf):
                g_of[(r2, c2)] = ng; came[(r2, c2)] = cur
                heapq.heappush(openh, (ng + h(r2, c2), ng, (r2, c2)))
    return None


class Planner:
    """
    Two layers. A* runs on a 4x-downsampled grid, only when the goal changes
    or the corridor trips -- it costs tens of ms and cannot run at 10 Hz. The
    corridor check is pure numpy over a few hundred cells and runs every tick.

    When the corridor trips the vehicle BRAKES TO A HOVER and replans; it does
    not swerve. With a 7 Hz lidar, tilt-gated mapping, and an airframe that
    must tilt (blinding the lidar) to accelerate sideways, a reactive dodge is
    how you fly into the thing you were avoiding.
    """

    def __init__(self, grid: OccupancyGrid2D, cfg: dict, factor: int = 4):
        v = cfg["vehicle"]
        self.grid = grid; self.f = factor
        self.inflate = v["radius"] + v["safety_margin"]
        self.decel, self.reaction = v["decel"], v["reaction"]

    def stopping_distance(self, speed):
        return speed * self.reaction + speed * speed / (2.0 * max(self.decel, 0.1))

    def max_safe_speed(self):
        usable = self.grid.max_range - self.inflate
        best = 0.3
        for s in np.arange(0.3, 8.0, 0.05):
            if self.stopping_distance(float(s)) <= usable:
                best = float(s)
            else:
                break
        return best

    def plan(self, start_xy, goal_xy):
        occ = self.grid.inflated(self.inflate)
        n = occ.shape[0] // self.f * self.f
        coarse = occ[:n, :n].reshape(n // self.f, self.f, n // self.f, self.f).any(axis=(1, 3))
        sc, sr = self.grid.world_to_cell(*start_xy)
        gc, gr = self.grid.world_to_cell(*goal_xy)
        start = (int(sr) // self.f, int(sc) // self.f)
        goal = (int(gr) // self.f, int(gc) // self.f)
        if 0 <= start[0] < coarse.shape[0] and 0 <= start[1] < coarse.shape[1]:
            coarse[start] = False                    # never trap ourselves
        cells = astar(coarse, start, goal)
        if cells is None:
            return None
        cells = self._simplify(cells, coarse)
        out = []
        for r, c in cells:
            x, y = self.grid.cell_to_world(c * self.f + self.f // 2, r * self.f + self.f // 2)
            out.append((float(x), float(y)))
        out[0] = tuple(start_xy); out[-1] = tuple(goal_xy)
        return out

    def _simplify(self, path, blocked):
        if len(path) < 3:
            return path
        out = [path[0]]; i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1 and not self._clear(blocked, path[i], path[j]):
                j -= 1
            out.append(path[j]); i = j
        return out

    @staticmethod
    def _clear(blocked, a, b):
        n = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
        if n == 0:
            return True
        rs = np.linspace(a[0], b[0], n + 1).round().astype(int)
        cs = np.linspace(a[1], b[1], n + 1).round().astype(int)
        return not blocked[rs, cs].any()

    def corridor_clear(self, x, y, heading, look, half_width=None):
        hw = self.inflate if half_width is None else half_width
        occ = self.grid.obstacles()
        ds = np.linspace(0.25, look, max(2, int(look / self.grid.res)))
        offs = np.linspace(-hw, hw, max(3, int(2 * hw / self.grid.res)))
        dd, oo = np.meshgrid(ds, offs)
        px = x + dd * math.cos(heading) - oo * math.sin(heading)
        py = y + dd * math.sin(heading) + oo * math.cos(heading)
        col, row = self.grid.world_to_cell(px.ravel(), py.ravel())
        ins = (col >= 0) & (col < self.grid.n) & (row >= 0) & (row < self.grid.n)
        if not ins.any():
            return True
        return not occ[row[ins], col[ins]].any()


def coverage_path(polygon, spacing, start_xy=None):
    """Boustrophedon sweep along the polygon's longer axis (fewer, longer legs)."""
    pts = np.asarray(polygon, dtype=float)
    if len(pts) < 3:
        return []
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = hi - lo
    sweep_x = span[0] >= span[1]
    legs = []

    def span_at(value, axis):
        other = 1 - axis; hits = []
        for i in range(len(pts)):
            p, q = pts[i], pts[(i + 1) % len(pts)]
            if (p[axis] - value) * (q[axis] - value) > 0 or p[axis] == q[axis]:
                continue
            u = (value - p[axis]) / (q[axis] - p[axis])
            hits.append(p[other] + u * (q[other] - p[other]))
        return (min(hits), max(hits)) if len(hits) >= 2 else None

    if sweep_x:
        for i, yv in enumerate(np.arange(lo[1] + spacing / 2, hi[1], spacing)):
            sp = span_at(yv, 1)
            if sp is None:
                continue
            a, b = sp if i % 2 == 0 else sp[::-1]
            legs += [(a, yv), (b, yv)]
    else:
        for i, xv in enumerate(np.arange(lo[0] + spacing / 2, hi[0], spacing)):
            sp = span_at(xv, 0)
            if sp is None:
                continue
            a, b = sp if i % 2 == 0 else sp[::-1]
            legs += [(xv, a), (xv, b)]
    if start_xy and legs:
        d0 = math.dist(start_xy, legs[0]); d1 = math.dist(start_xy, legs[-1])
        if d1 < d0:
            legs = legs[::-1]
    return legs


# ===========================================================================
# WORLD PROCESS  --  the one CPU-bound child. SLAM + map + planning together,
# because splitting them would push a 230 KB grid across a process boundary.
# ===========================================================================

def world_process(cfg, scan_q, plan_req_q, plan_res_q, state: SharedState, stop_evt):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s world: %(message)s",
                        datefmt="%H:%M:%S")
    wlog = logging.getLogger("world")
    try:
        os.sched_setaffinity(0, {1})            # keep off the I/O core
    except (AttributeError, OSError):
        pass

    grid = OccupancyGrid2D(cfg)
    matcher = ScanMatcher(cfg)
    planner = Planner(grid, cfg)
    mcfg = cfg["map"]
    wlog.info("started | grid %dx%d @ %.2f m | max safe speed %.2f m/s",
              grid.n, grid.n, grid.res, planner.max_safe_speed())

    last_grid_push = 0.0
    latencies = []

    while not stop_evt.is_set():
        state.world_alive.value = time.monotonic()

        # ---- planning requests (event-driven, not every tick) ----
        try:
            req = plan_req_q.get_nowait()
        except queue.Empty:
            req = None
        if req is not None:
            kind = req.get("kind")
            try:
                if kind == "goto":
                    path = planner.plan(req["start"], req["goal"])
                    plan_res_q.put({"kind": "path", "path": path,
                                    "goal": req["goal"], "seq": req.get("seq", 0)})
                elif kind == "coverage":
                    legs = coverage_path(req["polygon"], req["spacing"], req.get("start"))
                    plan_res_q.put({"kind": "coverage", "path": legs, "seq": req.get("seq", 0)})
                elif kind == "corridor":
                    ok = planner.corridor_clear(req["x"], req["y"], req["heading"], req["look"])
                    plan_res_q.put({"kind": "corridor", "clear": bool(ok),
                                    "seq": req.get("seq", 0)})
            except Exception as exc:
                wlog.error("planning failed: %s", exc)
                plan_res_q.put({"kind": kind, "path": None, "error": str(exc),
                                "seq": req.get("seq", 0)})

        # ---- scans: SLAM then mapping ----
        try:
            t_scan, ang, rng, att = scan_q.get(timeout=0.05)
        except queue.Empty:
            continue

        try:
            res = matcher.update(t_scan, ang, rng, mcfg["min_range"], mcfg["max_range"])
            if res is None:
                continue
            sx, sy, syaw, fitness = res
            roll, pitch, agl = att

            p = Pose(t=t_scan, x=sx, y=sy, z=agl,
                     vx=float(matcher.vel[0]), vy=float(matcher.vel[1]),
                     roll=roll, pitch=pitch, yaw=syaw, rel_alt=agl, valid=True)
            state.write_pose(p, fitness)
            state.slam_stamp.value = t_scan

            grid.integrate(ang, rng, sx, sy, syaw, math.hypot(roll, pitch), agl)

            latencies.append(time.monotonic() - t_scan)
            if len(latencies) > 200:
                latencies.pop(0)

            now = time.monotonic()
            if now - last_grid_push > 0.25:
                state.write_grid(grid.probability().astype(np.float32),
                                 grid.res, grid.origin[0], grid.origin[1])
                last_grid_push = now
        except Exception as exc:
            wlog.error("scan processing failed: %s", exc, exc_info=False)

    if latencies:
        p50 = float(np.percentile(latencies, 50)); p95 = float(np.percentile(latencies, 95))
        wlog.info("scan->pose latency p50 %.0f ms p95 %.0f ms  "
                  "<-- set ArduPilot VISO_DELAY_MS to about %.0f",
                  p50 * 1000, p95 * 1000, p50 * 1000)
    wlog.info("matches=%d rejects=%d | map updates=%d tilt-rej=%d floor-rej=%d",
              matcher.matches, matcher.rejects, grid.updates, grid.rej_tilt, grid.rej_ground)


# ===========================================================================
# THREAD: LIDAR
# ===========================================================================

LIDAR_PRESETS = {   # baud, single_ch, sample_khz, max_range, hz, motor_dtr
    "X2L": (115200, True, 3, 8.0, 7.0, True),
    "X2":  (115200, True, 3, 8.0, 7.0, True),
    "X4":  (128000, False, 5, 10.0, 10.0, False),
    "G4":  (230400, False, 9, 16.0, 10.0, False),
}


class LidarThread(threading.Thread):
    """
    A scan is a REVOLUTION, not an instant. At 7 Hz its first and last points
    are 140 ms apart, so we stamp the MIDPOINT -- that halves worst-case pose
    error versus stamping on arrival, and the timestamp feeds straight through
    to VISION_POSITION_ESTIMATE where latency becomes position error.
    """

    def __init__(self, cfg, scan_q, att_getter, stop_evt):
        super().__init__(daemon=True, name="lidar_rx")
        self.cfg = cfg; self.q = scan_q; self.att = att_getter; self.stop = stop_evt
        m = cfg["lidar"]["model"].upper()
        (self.baud, self.single, self.sample,
         self.max_range, self.hz, self.dtr) = LIDAR_PRESETS[m]
        self.model = m; self.port = cfg["lidar"]["port"]
        self.scans = 0; self.dropped = 0; self.last_scan = 0.0
        self.error: str | None = None
        self._laser = None

    def _open(self):
        import ydlidar
        ydlidar.os_init()
        laser = ydlidar.CYdLidar(); S = laser.setlidaropt
        S(ydlidar.LidarPropSerialPort, self.port)
        S(ydlidar.LidarPropSerialBaudrate, self.baud)
        S(ydlidar.LidarPropLidarType, ydlidar.TYPE_TRIANGLE)
        S(ydlidar.LidarPropDeviceType, ydlidar.YDLIDAR_TYPE_SERIAL)
        S(ydlidar.LidarPropScanFrequency, float(self.hz))
        S(ydlidar.LidarPropSampleRate, int(self.sample))
        S(ydlidar.LidarPropSingleChannel, bool(self.single))   # X2L: MUST be True
        S(ydlidar.LidarPropMaxAngle, 180.0); S(ydlidar.LidarPropMinAngle, -180.0)
        S(ydlidar.LidarPropMaxRange, float(self.max_range))
        S(ydlidar.LidarPropMinRange, 0.10)
        for name, val in (("LidarPropSupportMotorDtrCtrl", self.dtr),
                          ("LidarPropReversion", False), ("LidarPropInverted", True),
                          ("LidarPropAutoReconnect", True),
                          ("LidarPropFixedResolution", True)):
            prop = getattr(ydlidar, name, None)
            if prop is not None:
                S(prop, bool(val))
        prop = getattr(ydlidar, "LidarPropAbnormalCheckCount", None)
        if prop is not None:
            S(prop, 4)
        if not laser.initialize():
            raise RuntimeError(f"initialize() failed on {self.port} @ {self.baud}")
        if not laser.turnOn():
            laser.disconnecting()
            raise RuntimeError("turnOn() failed -- almost always power, not data")
        return laser

    def run(self):
        import ydlidar
        backoff = 1.0
        while not self.stop.is_set():
            try:
                if self._laser is None:
                    self._laser = self._open()
                    log.info("lidar %s streaming on %s", self.model, self.port)
                    backoff = 1.0
                raw = ydlidar.LaserScan()
                while not self.stop.is_set():
                    t_end = time.monotonic()
                    if not self._laser.doProcessSimple(raw):
                        time.sleep(0.005)
                        continue
                    pts = raw.points; n = len(pts)
                    if n == 0:
                        continue
                    ang = np.fromiter((p.angle for p in pts), np.float32, n)
                    rng = np.fromiter((p.range for p in pts), np.float32, n)
                    scan_time = float(raw.config.scan_time) or (1.0 / self.hz)
                    t_mid = t_end - scan_time * 0.5
                    self.scans += 1; self.last_scan = time.monotonic()
                    try:
                        if self.q.full():
                            self.q.get_nowait(); self.dropped += 1
                        self.q.put_nowait((t_mid, ang, rng, self.att()))
                    except Exception:
                        self.dropped += 1
            except Exception as exc:
                self.error = str(exc)
                log.error("lidar fault: %s -- retrying in %.0fs", exc, backoff)
                try:
                    if self._laser:
                        self._laser.turnOff(); self._laser.disconnecting()
                except Exception:
                    pass
                self._laser = None
                self.stop.wait(backoff)
                backoff = min(backoff * 2, 15.0)      # recover, don't crash

    def close(self):
        if self._laser:
            try:
                self._laser.turnOff()
            finally:
                self._laser.disconnecting()
            self._laser = None


# ===========================================================================
# THREAD: MAVLINK  --  FC state in, vision pose out, GCS proxy
# ===========================================================================

class MavlinkLink:
    def __init__(self, cfg, state: SharedState, sitl=False):
        self.cfg = cfg; self.state = state; self.sitl = sitl
        self.master = None; self.proxy = None
        self.tgt_sys = 1; self.tgt_comp = 1
        self.roll = self.pitch = self.yaw_enu = 0.0
        self.rel_alt = 0.0; self.armed = False; self.mode = "?"
        self.gps_fix = 0; self.sats = 0; self.batt = -1.0
        self.fc_vx = self.fc_vy = 0.0
        self.last_msg = 0.0
        self.vision_sent = 0; self.reset_counter = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def connect(self, timeout=30.0):
        from pymavlink import mavutil
        dev = "udp:127.0.0.1:14550" if self.sitl else self.cfg["mavlink"]["device"]
        log.info("MAVLink connecting to %s", dev)
        if dev.startswith(("udp", "tcp")):
            self.master = mavutil.mavlink_connection(dev, source_system=1)
        else:
            self.master = mavutil.mavlink_connection(dev, baud=self.cfg["mavlink"]["baud"],
                                                     source_system=1)
        if self.master.wait_heartbeat(timeout=timeout) is None:
            raise TimeoutError(f"no heartbeat from {dev}")
        self.tgt_sys = self.master.target_system
        self.tgt_comp = self.master.target_component
        log.info("FC system %d component %d", self.tgt_sys, self.tgt_comp)
        self._request_streams()
        if self.cfg["mavlink"]["proxy"]:
            try:
                self.proxy = mavutil.mavlink_connection(self.cfg["mavlink"]["proxy_out"],
                                                        source_system=1, input=False)
                log.info("MAVLink proxy -> %s", self.cfg["mavlink"]["proxy_out"])
            except Exception as exc:
                log.warning("proxy unavailable: %s", exc)

    def _request_streams(self):
        """
        Only what we need. pymavlink parsing is pure Python and HOLDS the GIL;
        streaming everything is the easiest way to starve the other threads.
        """
        from pymavlink import mavutil
        for msg_id, hz in ((mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, 30),
                           (mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, 10),
                           (mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 4),
                           # GPS_RAW_INT carries fix_type / satellites / HDOP.
                           # Without requesting it, gps_fix stays 0 forever and
                           # the preflight GPS check can never pass outdoors.
                           (mavutil.mavlink.MAVLINK_MSG_ID_GPS_RAW_INT, 2),
                           (mavutil.mavlink.MAVLINK_MSG_ID_SYS_STATUS, 1)):
            self.master.mav.command_long_send(
                self.tgt_sys, self.tgt_comp,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                msg_id, int(1e6 / hz), 0, 0, 0, 0, 0)
            time.sleep(0.02)

    def attitude(self):
        return (self.roll, self.pitch, self.rel_alt)

    # ---- receive ----
    def _rx_loop(self):
        from pymavlink import mavutil
        while not self._stop.is_set():
            try:
                msg = self.master.recv_match(blocking=True, timeout=0.5)
            except Exception as exc:
                log.error("mavlink recv: %s", exc); time.sleep(0.5); continue
            if msg is None:
                continue
            self.last_msg = time.monotonic()
            t = msg.get_type()
            if t == "ATTITUDE":
                self.roll, self.pitch = msg.roll, msg.pitch
                self.yaw_enu = yaw_ned_to_enu(msg.yaw)
            elif t == "GLOBAL_POSITION_INT":
                self.rel_alt = msg.relative_alt / 1000.0
            elif t == "LOCAL_POSITION_NED":
                self.fc_vx, self.fc_vy = msg.vy, msg.vx      # NED -> ENU
            elif t == "GPS_RAW_INT":
                self.gps_fix, self.sats = msg.fix_type, msg.satellites_visible
            elif t == "SYS_STATUS":
                self.batt = msg.battery_remaining
            elif t == "HEARTBEAT":
                self.armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                try:
                    self.mode = mavutil.mode_string_v10(msg)
                except Exception:
                    self.mode = str(msg.custom_mode)
            if self.proxy is not None:
                try:
                    self.proxy.mav.srcSystem = msg.get_srcSystem()
                    self.proxy.write(msg.get_msgbuf())
                except Exception:
                    pass

    # ---- vision pose out ----
    def _vision_loop(self):
        """
        The lidar produces pose at 7 Hz; EKF3's ExternalNav prefers >=10 Hz.
        We extrapolate with the FC's own velocity between scans and publish at
        12 Hz. The timestamp stays anchored to the SLAM solution's scan time,
        never to 'now' -- stamping 'now' would inject pipeline latency as
        position error and EKF3 would read it as real motion.
        """
        from pymavlink import mavutil
        period = 1.0 / self.cfg["vision"]["send_hz"]
        while not self._stop.is_set():
            time.sleep(period)
            if not self.cfg["vision"]["enabled"]:
                continue
            p, fitness = self.state.read_pose()
            if p is None or not p.valid:
                continue
            age = time.monotonic() - p.t
            if age > 1.0:
                continue
            ex = p.x + self.fc_vx * age          # extrapolate to the send instant
            ey = p.y + self.fc_vy * age
            n, e, d = enu_to_ned(ex, ey, p.z)
            # covariance scaled by ICP residual: a poor match tells EKF3 to
            # trust this less rather than silently feeding it a bad fix
            var = max(0.01, min(1.0, fitness * fitness * 4.0))
            cov = [float("nan")] * 21
            cov[0] = var; cov[6] = var; cov[11] = var * 4.0
            try:
                with self._lock:
                    self.master.mav.vision_position_estimate_send(
                        int(p.t * 1e6) & 0xFFFFFFFFFFFFFFFF,
                        n, e, d, 0.0, 0.0, 0.0, cov, self.reset_counter)
                self.vision_sent += 1
            except Exception as exc:
                log.debug("vision send failed: %s", exc)

    def send_obstacle_distance(self, angles, ranges, tilt, max_tilt):
        """72 sectors. Mission Planner and QGC draw it natively; ArduPilot
        also consumes it for its own avoidance (PRX1_TYPE=2) -- a safety layer
        that survives this script crashing."""
        from pymavlink import mavutil
        if tilt > max_tilt:
            return
        dist = np.full(72, 65535, np.uint16)
        ok = np.isfinite(ranges) & (ranges > 0.15) & (ranges < 8.0)
        if ok.any():
            a = np.degrees(angles[ok]) % 360.0
            r_cm = np.clip(ranges[ok] * 100.0, 15, 800).astype(np.uint16)
            np.minimum.at(dist, (a / 5).astype(np.int32) % 72, r_cm)
        try:
            with self._lock:
                self.master.mav.obstacle_distance_send(
                    int(time.time() * 1e6) & 0xFFFFFFFFFFFFFFFF,
                    mavutil.mavlink.MAV_DISTANCE_SENSOR_LASER,
                    dist.tolist(), 5, 15, 800, 0.0, 0.0,
                    mavutil.mavlink.MAV_FRAME_BODY_FRD)
        except Exception:
            pass

    # ---- commands ----
    def _cmd(self, command, *params):
        from pymavlink import mavutil
        p = list(params) + [0.0] * (7 - len(params))
        with self._lock:
            self.master.mav.command_long_send(self.tgt_sys, self.tgt_comp,
                                              command, 0, *p[:7])
        ack = self.master.recv_match(type="COMMAND_ACK", blocking=True, timeout=3.0)
        return ack is not None and ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED

    def set_mode(self, mode="GUIDED"):
        mapping = self.master.mode_mapping()
        if not mapping or mode not in mapping:
            return False
        with self._lock:
            self.master.set_mode(mapping[mode])
        for _ in range(30):
            time.sleep(0.1)
            if self.mode.upper() == mode.upper():
                return True
        return False

    def arm(self):
        from pymavlink import mavutil
        return self._cmd(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1)

    def disarm(self, force=False):
        from pymavlink import mavutil
        return self._cmd(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                         0, 21196 if force else 0)

    def takeoff(self, alt):
        from pymavlink import mavutil
        return self._cmd(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, alt)

    def land(self):
        from pymavlink import mavutil
        return self._cmd(mavutil.mavlink.MAV_CMD_NAV_LAND)

    def goto(self, x, y, z, yaw=None):
        from pymavlink import mavutil
        mask = 0b0000_1111_1111_1000
        if yaw is None:
            mask |= 0b0000_1000_0000_0000
        n, e, d = enu_to_ned(x, y, z)
        with self._lock:
            self.master.mav.set_position_target_local_ned_send(
                int(time.monotonic() * 1000) & 0xFFFFFFFF, self.tgt_sys, self.tgt_comp,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED, mask,
                n, e, d, 0, 0, 0, 0, 0, 0,
                0.0 if yaw is None else wrap_pi(math.pi / 2 - yaw), 0)

    def hold(self):
        from pymavlink import mavutil
        with self._lock:
            self.master.mav.set_position_target_local_ned_send(
                int(time.monotonic() * 1000) & 0xFFFFFFFF, self.tgt_sys, self.tgt_comp,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED, 0b0000_1111_1100_0111,
                0, 0, 0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0)

    def start(self):
        self._stop.clear()
        for fn, name in ((self._rx_loop, "mavlink_rx"), (self._vision_loop, "vision_tx")):
            th = threading.Thread(target=fn, daemon=True, name=name)
            th.start(); self._threads.append(th)

    def stop(self):
        self._stop.set()
        for th in self._threads:
            th.join(timeout=2.0)


# ===========================================================================
# THREAD: TELEMETRY  --  fire and forget, bounded, never blocks
# ===========================================================================

class TelemetryThread(threading.Thread):
    """
    The property that matters is not bandwidth (~34 KB/s), it is that a GCS
    which sleeps, drops off WiFi or hangs must never stall the flight loop.
    ZMQ PUB with a bounded queue and non-blocking sends drops instead.

    SNDHWM is 8, not 1. With a depth of one, whichever topic publishes first
    each cycle fills the only slot and every other topic is dropped forever --
    you receive state and nothing else.
    """

    def __init__(self, cfg, state: SharedState, status_getter, stop_evt):
        super().__init__(daemon=True, name="telemetry")
        self.cfg = cfg["telemetry"]; self.state = state
        self.status = status_getter; self.stop_evt = stop_evt
        self.sock = None; self.sent = 0; self.dropped = 0
        self.enabled = self.cfg["enabled"]
        try:
            import zmq, msgpack
            self._zmq, self._mp = zmq, msgpack
        except ImportError:
            log.warning("pyzmq/msgpack missing -- telemetry off")
            self.enabled = False

    def run(self):
        if not self.enabled:
            return
        zmq = self._zmq
        ctx = zmq.Context.instance()
        self.sock = ctx.socket(zmq.PUB)
        self.sock.setsockopt(zmq.SNDHWM, 8)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.bind(self.cfg["bind"])
        log.info("telemetry on %s", self.cfg["bind"])

        t_state = t_map = 0.0
        sp, mp_ = 1.0 / self.cfg["state_hz"], 1.0 / self.cfg["map_hz"]
        while not self.stop_evt.is_set():
            now = time.monotonic()
            if now - t_state >= sp:
                t_state = now
                p, fit = self.state.read_pose()
                if p is not None:
                    st = self.status()
                    st.update({"t": p.t, "x": p.x, "y": p.y, "z": p.z,
                               "vx": p.vx, "vy": p.vy, "yaw": p.yaw,
                               "roll": p.roll, "pitch": p.pitch,
                               "rel_alt": p.rel_alt, "fitness": fit})
                    self._send("state", st)
            if now - t_map >= mp_:
                t_map = now
                g, meta = self.state.read_grid()
                if g is not None and meta[1] > 0:
                    u8 = (g * 255).astype(np.uint8)
                    self._send("map", {"t": meta[0], "n": int(g.shape[0]),
                                       "res": meta[1], "origin_x": meta[2],
                                       "origin_y": meta[3], "codec": "zlib-u8"},
                               zlib.compress(u8.tobytes(), 1))
            time.sleep(0.02)
        if self.sock:
            self.sock.close(linger=0)

    def _send(self, topic, header, payload=b""):
        try:
            parts = [topic.encode(), self._mp.packb(header, use_bin_type=True)]
            if payload:
                parts.append(payload)
            self.sock.send_multipart(parts, flags=self._zmq.NOBLOCK)
            self.sent += 1
        except Exception:
            self.dropped += 1


# ===========================================================================
# ORCHESTRATOR  --  the single entry point
# ===========================================================================

class Orchestrator:
    def __init__(self, cfg, sitl=False):
        self.cfg = cfg; self.sitl = sitl
        self.state_name = FlightState.BOOT
        self.abort_reason = Abort.NONE
        self.entered = time.monotonic()

        grid_n = int(round(cfg["map"]["size_m"] / cfg["map"]["resolution"]))
        self.shared = SharedState(grid_n)
        self.scan_q = mp.Queue(maxsize=3)
        self.plan_req = mp.Queue(maxsize=4)
        self.plan_res = mp.Queue(maxsize=4)
        self.stop_evt = mp.Event()

        self.mav = None; self.lidar = None; self.telem = None; self.world = None
        self.goal = None; self.path = []; self.leg = 0
        self.home = (0.0, 0.0); self.replans = 0
        self._plan_seq = 0; self._await_plan = False
        self._await_corridor = False; self._corridor_ok = True
        self._last_sp = 0.0; self._last_corr = 0.0
        self.max_speed = 1.0

    # ---- lifecycle ----
    def start(self):
        self.mav = MavlinkLink(self.cfg, self.shared, self.sitl)
        self.mav.connect()
        self.mav.start()

        self.world = mp.Process(target=world_process, name="world",
                                args=(self.cfg, self.scan_q, self.plan_req,
                                      self.plan_res, self.shared, self.stop_evt),
                                daemon=True)
        self.world.start()

        if self.cfg["lidar"]["enabled"]:
            self.lidar = LidarThread(self.cfg, self.scan_q, self.mav.attitude, self.stop_evt)
            self.lidar.start()

        self.telem = TelemetryThread(self.cfg, self.shared, self.status, self.stop_evt)
        self.telem.start()

        v = self.cfg["vehicle"]
        usable = self.cfg["map"]["max_range"] - v["radius"] - v["safety_margin"]
        best = 0.3
        for s in np.arange(0.3, 8.0, 0.05):
            if s * v["reaction"] + s * s / (2 * v["decel"]) <= usable:
                best = float(s)
            else:
                break
        self.max_speed = best
        if self.cfg["flight"]["cruise_speed"] > best:
            log.warning("cruise_speed %.2f exceeds what a %.1f m lidar can stop for; "
                        "clamping to %.2f m/s", self.cfg["flight"]["cruise_speed"],
                        self.cfg["map"]["max_range"], best)
            self.cfg["flight"]["cruise_speed"] = best
        self._set(FlightState.PREFLIGHT)

    def status(self):
        return {"state": self.state_name.name, "abort": self.abort_reason.name,
                "armed": self.mav.armed, "mode": self.mav.mode,
                "battery": self.mav.batt, "gps_fix": self.mav.gps_fix,
                "scans": self.lidar.scans if self.lidar else 0,
                "slam_age": time.monotonic() - self.shared.slam_stamp.value
                if self.shared.slam_stamp.value else 999.0,
                "vision_sent": self.mav.vision_sent}

    def _set(self, s, why=""):
        if s != self.state_name:
            log.info("STATE %s -> %s %s", self.state_name.name, s.name,
                     f"({why})" if why else "")
            self.state_name = s; self.entered = time.monotonic()

    def _el(self):
        return time.monotonic() - self.entered

    def _abort(self, reason, msg=""):
        if self.state_name in (FlightState.ABORT, FlightState.LANDED):
            return
        self.abort_reason = reason
        log.error("ABORT %s %s", reason.name, msg)
        try:
            self.mav.hold()
        except Exception:
            pass
        self._set(FlightState.ABORT, reason.name)

    # ---- watchdog ----
    def _watchdog(self, pose):
        lim = self.cfg["limits"]; fl = self.cfg["flight"]
        airborne = self.state_name in (FlightState.TAKEOFF, FlightState.NAVIGATE,
                                       FlightState.HOLD, FlightState.LANDING)

        if self.world is not None and not self.world.is_alive():
            self._abort(Abort.WORLD_DIED, "SLAM process exited")
            return False
        if airborne and time.monotonic() - self.shared.world_alive.value > 2.0:
            self._abort(Abort.WORLD_DIED, "SLAM process stalled")
            return False

        # Indoors, EK3_SRC1_POSXY=6 makes our pose the FC's ONLY position
        # source. A SLAM stall is a crash, not a degradation -- land at once.
        slam_age = (time.monotonic() - self.shared.slam_stamp.value
                    if self.shared.slam_stamp.value else 999.0)
        if airborne and self.cfg["vision"]["enabled"] and slam_age > lim["max_slam_age"]:
            self._abort(Abort.SLAM_LOST, f"pose {slam_age:.2f}s stale")
            return False
        if pose is None or not pose.valid:
            if airborne:
                self._abort(Abort.POSE_TIMEOUT)
                return False
            return True
        if airborne and self.lidar is not None:
            if time.monotonic() - self.lidar.last_scan > lim["max_scan_age"]:
                self._abort(Abort.LIDAR_TIMEOUT)
                return False
        if airborne:
            if math.hypot(pose.x - self.home[0], pose.y - self.home[1]) > fl["geofence_radius"]:
                self._abort(Abort.GEOFENCE); return False
            if self.mav.rel_alt > fl["max_alt"]:
                self._abort(Abort.GEOFENCE, "altitude"); return False
            if 0 <= self.mav.batt < fl["min_battery_pct"]:
                self._abort(Abort.LOW_BATTERY); return False
        if self.state_name in (FlightState.NAVIGATE, FlightState.HOLD) and not self.mav.armed:
            self._abort(Abort.FC_DISARMED); return False
        return True

    # ---- planning helpers ----
    def _request_plan(self, start, goal):
        self._plan_seq += 1
        try:
            self.plan_req.put_nowait({"kind": "goto", "start": start,
                                      "goal": goal, "seq": self._plan_seq})
            self._await_plan = True
        except Exception:
            pass

    def _request_corridor(self, x, y, heading, look):
        try:
            self.plan_req.put_nowait({"kind": "corridor", "x": x, "y": y,
                                      "heading": heading, "look": look,
                                      "seq": self._plan_seq})
            self._await_corridor = True
        except Exception:
            pass

    def _drain_plans(self):
        while True:
            try:
                r = self.plan_res.get_nowait()
            except queue.Empty:
                return
            if r["kind"] == "corridor":
                self._corridor_ok = r["clear"]; self._await_corridor = False
            elif r["kind"] in ("path", "coverage"):
                self._await_plan = False
                self.path = r.get("path") or []
                self.leg = 0
                if not self.path:
                    log.warning("planner returned no path")

    # ---- state handlers ----
    def _preflight(self, pose):
        checks = {
            "FC heartbeat": time.monotonic() - self.mav.last_msg < 3.0,
            "world process": self.world.is_alive(),
            "lidar scans": (not self.cfg["lidar"]["enabled"]) or
                           (self.lidar and self.lidar.scans > 5),
            "slam pose": pose is not None and pose.valid,
            "battery": self.mav.batt < 0 or self.mav.batt > self.cfg["flight"]["min_battery_pct"],
        }
        if self._el() < 1.0 or int(self._el()) % 3 == 0:
            for k, ok in checks.items():
                log.info("  preflight %-14s %s", k, "OK" if ok else "waiting")
        if not all(checks.values()):
            if self._el() > 45.0:
                self._abort(Abort.OPERATOR, "preflight never passed")
            return
        self.home = (pose.x, pose.y)
        log.info("home %.2f,%.2f | cruise %.2f m/s (cap %.2f)", *self.home,
                 self.cfg["flight"]["cruise_speed"], self.max_speed)
        self._set(FlightState.ARMING)

    def _arming(self):
        if not self.mav.set_mode("GUIDED"):
            self._abort(Abort.OPERATOR, "GUIDED refused"); return
        if not self.mav.arm():
            if self._el() > 12.0:
                self._abort(Abort.OPERATOR, "arm refused")
            return
        time.sleep(1.0)
        if not self.mav.takeoff(self.cfg["flight"]["takeoff_alt"]):
            self._abort(Abort.OPERATOR, "takeoff refused"); return
        self._set(FlightState.TAKEOFF)

    def _takeoff(self):
        tgt = self.cfg["flight"]["takeoff_alt"]
        if self.mav.rel_alt >= tgt * 0.95:
            self._set(FlightState.NAVIGATE); return
        if self._el() > 60.0:
            self._abort(Abort.OPERATOR, f"only {self.mav.rel_alt:.1f} m after 60 s")

    def _navigate(self, pose):
        if self.goal is None:
            self._set(FlightState.LANDING, "no goal"); return
        now = time.monotonic()
        cruise = self.cfg["flight"]["cruise_speed"]
        v = self.cfg["vehicle"]
        # look ahead for the speed we are ABOUT to reach, not the current one
        look = max(1.0, cruise * v["reaction"] + cruise * cruise / (2 * v["decel"]))
        if now - self._last_corr > 0.1 and not self._await_corridor:
            self._request_corridor(pose.x, pose.y, pose.yaw, look)
            self._last_corr = now
        if not self._corridor_ok:
            log.warning("obstacle within %.1f m -- braking", look)
            self.mav.hold(); self._set(FlightState.HOLD, "corridor blocked"); return

        if not self.path and not self._await_plan:
            self._request_plan((pose.x, pose.y), self.goal); return
        if not self.path:
            return
        if self.leg >= len(self.path):
            self._set(FlightState.LANDING, "goal reached"); return
        wx, wy = self.path[self.leg]
        if math.hypot(wx - pose.x, wy - pose.y) < self.cfg["flight"]["wp_tolerance"]:
            self.leg += 1; return
        if now - self._last_sp > 0.2:
            self.mav.goto(wx, wy, self.cfg["flight"]["takeoff_alt"],
                          math.atan2(wy - pose.y, wx - pose.x))
            self._last_sp = now

    def _hold(self, pose):
        self.mav.hold()
        if self._el() < 1.5:
            return
        if math.hypot(pose.vx, pose.vy) > 0.3:
            return
        if self._await_plan:
            return
        self.replans += 1
        if self.replans > self.cfg["limits"]["replan_attempts"]:
            self._abort(Abort.NO_PATH, f"{self.replans} replans failed"); return
        log.info("replanning (attempt %d)", self.replans)
        self.path = []; self._corridor_ok = True
        self._request_plan((pose.x, pose.y), self.goal)
        self._set(FlightState.NAVIGATE, "replanned")

    def _landing(self):
        if self._el() < 0.5:
            self.mav.land(); return
        if not self.mav.armed or self.mav.rel_alt < 0.2:
            self._set(FlightState.LANDED)
        elif self._el() > 90.0:
            log.error("landing timeout -- forcing disarm")
            self.mav.disarm(force=True); self._set(FlightState.LANDED)

    def _abort_state(self):
        if self._el() < 0.5:
            self.mav.land(); return
        if not self.mav.armed or self.mav.rel_alt < 0.2:
            self._set(FlightState.LANDED, "abort complete")

    # ---- main loop ----
    def tick(self):
        self._drain_plans()
        pose, _ = self.shared.read_pose()
        if self.state_name not in (FlightState.BOOT, FlightState.LANDED):
            if not self._watchdog(pose):
                pose, _ = self.shared.read_pose()

        s = self.state_name
        if s == FlightState.PREFLIGHT:  self._preflight(pose)
        elif s == FlightState.ARMING:   self._arming()
        elif s == FlightState.TAKEOFF:  self._takeoff()
        elif s == FlightState.NAVIGATE and pose: self._navigate(pose)
        elif s == FlightState.HOLD and pose:     self._hold(pose)
        elif s == FlightState.LANDING:  self._landing()
        elif s == FlightState.ABORT:    self._abort_state()
        elif s == FlightState.LANDED:   return False
        return True

    def run(self, goal):
        self.goal = goal
        try:
            while self.tick():
                time.sleep(0.1)
        except KeyboardInterrupt:
            log.warning("operator interrupt")
            self._abort(Abort.OPERATOR)
            deadline = time.monotonic() + 90
            while self.state_name != FlightState.LANDED and time.monotonic() < deadline:
                self.tick(); time.sleep(0.1)
        finally:
            self.shutdown()
        return 0 if self.abort_reason == Abort.NONE else 1

    def shutdown(self):
        log.info("shutting down")
        self.stop_evt.set()
        if self.world is not None:
            self.world.join(timeout=4.0)
            if self.world.is_alive():
                self.world.terminate()
        if self.lidar is not None:
            self.lidar.join(timeout=2.0); self.lidar.close()
            log.info("lidar: %d scans, %d dropped", self.lidar.scans, self.lidar.dropped)
        if self.mav is not None:
            log.info("vision poses sent: %d", self.mav.vision_sent)
            self.mav.stop()
        if self.telem is not None:
            self.telem.join(timeout=2.0)
            log.info("telemetry: %d sent, %d dropped", self.telem.sent, self.telem.dropped)
        self.shared.close()


def main():
    ap = argparse.ArgumentParser(description="Disaster rescue drone -- Pi companion")
    ap.add_argument("--config")
    ap.add_argument("--sitl", action="store_true")
    ap.add_argument("--goal", nargs=2, type=float, metavar=("X", "Y"), default=[5.0, 0.0])
    ap.add_argument("--no-lidar", action="store_true")
    ap.add_argument("--no-vision", action="store_true",
                    help="do not send VISION_POSITION_ESTIMATE to the FC")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    if not HAVE_SCIPY:
        log.error("scipy is REQUIRED: its cKDTree makes ICP 11x faster "
                  "(4 ms vs 46 ms per match). sudo apt install python3-scipy")
        return 2

    cfg = dict(CONFIG)
    if args.config:
        import yaml
        with open(args.config) as fh:
            cfg = deep_merge(cfg, yaml.safe_load(fh) or {})
    if args.no_lidar:
        cfg["lidar"]["enabled"] = False
    if args.no_vision:
        cfg["vision"]["enabled"] = False

    mp.set_start_method("fork", force=True)
    orch = Orchestrator(cfg, sitl=args.sitl)
    signal.signal(signal.SIGTERM, lambda *_: orch._abort(Abort.OPERATOR, "SIGTERM"))
    orch.start()
    return orch.run(tuple(args.goal))


if __name__ == "__main__":
    sys.exit(main())
