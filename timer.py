"""
timer.py
Per-person working hour accumulator.

Tracks:
- Total on-screen time per identity
- Individual work sessions (continuous presence intervals)
- Gaps and break detection
- Hourly activity heatmaps
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
import time, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


@dataclass
class WorkSession:
    """A single continuous work session for one person."""
    person_id:  str
    start_frame: int
    end_frame:   int
    start_time:  float  # seconds from video start
    end_time:    float
    duration:    float  # seconds

    @property
    def duration_hours(self) -> float:
        return round(self.duration / 3600, 4)

    @property
    def duration_hms(self) -> str:
        h = int(self.duration // 3600)
        m = int((self.duration % 3600) // 60)
        s = int(self.duration % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"


class WorkingHourEstimator:
    """
    Accumulates on-screen durations and detects work sessions.
    Call update() every processed frame with the set of visible person IDs.
    """

    def __init__(self, fps: float = None):
        self.fps = fps or config.VIDEO_FPS / max(1, config.FRAME_SKIP)
        self.frame_number   = 0

        # {person_id → total seconds on screen}
        self.total_seconds: dict[str, float] = defaultdict(float)

        # {person_id → last frame seen}
        self._last_frame: dict[str, int] = {}

        # {person_id → session start frame}
        self._session_start: dict[str, int] = {}
        self._session_start_time: dict[str, float] = {}

        # Completed sessions
        self.sessions: list[WorkSession] = []

        # Frame-level presence log {person_id → list of (frame, time_sec)}
        self._presence_log: dict[str, list] = defaultdict(list)

        # Hourly bins: {person_id → np.array of 24 buckets}
        self._hourly_bins: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(24))

        print(f"[Timer] Working hour estimator ready. Effective FPS = {self.fps:.2f}")

    # ─── Public API ───────────────────────────────────────────────────────────

    def update(self, visible_person_ids: set[str]):
        """
        Call once per processed frame.
        visible_person_ids: set of person labels currently on screen.
        """
        current_time = self.frame_number / self.fps
        self.frame_number += 1

        for pid in visible_person_ids:
            self._presence_log[pid].append((self.frame_number, current_time))

            # Start new session if not active
            if pid not in self._session_start:
                self._session_start[pid]      = self.frame_number
                self._session_start_time[pid] = current_time

            # Accumulate presence
            gap = self.frame_number - self._last_frame.get(pid, self.frame_number)
            if gap <= int(config.MAX_GAP_SECONDS * self.fps):
                self.total_seconds[pid] += 1.0 / self.fps
            else:
                # Gap too large — close old session, start new one
                self._close_session(pid)
                self._session_start[pid]      = self.frame_number
                self._session_start_time[pid] = current_time

            self._last_frame[pid] = self.frame_number

            # Hourly bin
            hour = int(current_time // 3600) % 24
            self._hourly_bins[pid][hour] += 1.0 / self.fps

        # Close sessions for people who left frame
        gone = set(self._session_start.keys()) - visible_person_ids
        for pid in gone:
            last = self._last_frame.get(pid, 0)
            gap_secs = (self.frame_number - last) / self.fps
            if gap_secs > config.MAX_GAP_SECONDS:
                self._close_session(pid)

    def get_report(self) -> pd.DataFrame:
        """Return a DataFrame with per-person working hour statistics."""
        rows = []
        for pid in sorted(set(list(self.total_seconds.keys()) +
                               [s.person_id for s in self.sessions])):
            total_secs = self.total_seconds.get(pid, 0.0)
            if total_secs < config.MIN_PRESENCE_SECONDS:
                continue
            person_sessions = [s for s in self.sessions if s.person_id == pid]
            rows.append({
                "Person ID":          pid,
                "Total Hours":        round(total_secs / 3600, 2),
                "Total (HH:MM:SS)":   _seconds_to_hms(total_secs),
                "Num Sessions":       len(person_sessions),
                "Longest Session (s)": max((s.duration for s in person_sessions), default=0),
                "First Seen (s)":     min((s.start_time for s in person_sessions), default=0),
                "Last Seen (s)":      max((s.end_time for s in person_sessions), default=0),
            })
        return pd.DataFrame(rows)

    def get_hourly_matrix(self) -> pd.DataFrame:
        """
        Returns a DataFrame (persons × 24 hours) with seconds-in-hour values.
        Useful for heatmap visualization.
        """
        if not self._hourly_bins:
            return pd.DataFrame()
        data = {pid: arr for pid, arr in self._hourly_bins.items()}
        df = pd.DataFrame(data, index=[f"Hour {h}" for h in range(24)]).T
        return df

    def get_sessions_df(self) -> pd.DataFrame:
        """Return all completed sessions as a DataFrame."""
        if not self.sessions:
            return pd.DataFrame()
        return pd.DataFrame([{
            "Person ID":    s.person_id,
            "Start (s)":    round(s.start_time, 1),
            "End (s)":      round(s.end_time, 1),
            "Duration (s)": round(s.duration, 1),
            "Duration":     s.duration_hms,
            "Start Frame":  s.start_frame,
            "End Frame":    s.end_frame,
        } for s in self.sessions])

    def get_timeline(self) -> dict:
        """Return presence log suitable for Gantt-style visualisation."""
        return {pid: times for pid, times in self._presence_log.items()}

    def finalize(self):
        """Call at end of video to close all open sessions."""
        for pid in list(self._session_start.keys()):
            self._close_session(pid)

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _close_session(self, pid: str):
        if pid not in self._session_start:
            return
        start_frame = self._session_start.pop(pid)
        start_time  = self._session_start_time.pop(pid)
        end_frame   = self._last_frame.get(pid, start_frame)
        end_time    = end_frame / self.fps
        duration    = end_time - start_time

        if duration >= config.MIN_PRESENCE_SECONDS:
            self.sessions.append(WorkSession(
                person_id   = pid,
                start_frame = start_frame,
                end_frame   = end_frame,
                start_time  = start_time,
                end_time    = end_time,
                duration    = duration
            ))


def _seconds_to_hms(secs: float) -> str:
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = int(secs % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
