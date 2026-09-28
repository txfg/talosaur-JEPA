"""SEARCH -> ACQUIRE -> TRACK <-> FILM -> LOST -> SEARCH, RELEASE, plus recording decisions.

* SEARCH: slow scan; "animal present" needs ``acquire_n`` of the last ``acquire_m`` frames
  (frame probability or heatmap peak above threshold) to leave SEARCH - one bright speck
  does not trigger anything. Animals already filmed are filtered out before this (encounters.py).
* ACQUIRE: the tracker must confirm the target (``confirm_hits`` updates) -> TRACK, else back to
  SEARCH after ``acquire_timeout_s`` (the detection window is cleared, so re-acquiring needs
  ``acquire_n`` fresh detections).
* TRACK: servo on the target and approach to the stand-off size. Entering TRACK starts an
  *encounter* and the recording (the pre-roll buffer keeps the approach).
* FILM: target centred and at a good size for ``film_hold_s`` -> hold framing (no approach).
* LOST: tracker lost the target -> turn toward the last bearing for ``lost_timeout_s``; re-acquire
  -> TRACK (same encounter), else SEARCH (encounter ends, reason "lost").
* RELEASE: the caller decided to leave the animal (``release=<reason>``: enough footage, it fled,
  no good shot, or the time budget - see encounters.py) -> back off, turn away and swim on for
  ``release_s`` while ignoring detections, then SEARCH for another animal (the encounter ends with
  that reason).

Recording stops ``postroll_s`` after leaving TRACK/FILM/LOST, so RELEASE also ends the recording.
Events: ``encounter_start`` / ``encounter_end`` (see ``end_reason``), ``start_recording`` /
``stop_recording``, ``state:<NAME>`` - in that order within one frame.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

from talosaur.guidance.heatmap import Target
from talosaur.guidance.tracker import TrackState


class State(str, Enum):
    SEARCH = "SEARCH"
    ACQUIRE = "ACQUIRE"
    TRACK = "TRACK"
    FILM = "FILM"
    LOST = "LOST"
    RELEASE = "RELEASE"


ENGAGED = (State.TRACK, State.FILM, State.LOST)


@dataclass
class FSMConfig:
    frame_on: float = 0.6
    heat_on: float = 0.6
    acquire_n: int = 3
    acquire_m: int = 5
    acquire_timeout_s: float = 3.0
    film_center_deg: float = 8.0
    film_size: tuple[float, float] = (0.12, 0.40)
    film_hold_s: float = 1.0
    lost_timeout_s: float = 4.0
    postroll_s: float = 3.0
    max_record_s: float = 300.0
    release_s: float = 12.0  # length of the move-on manoeuvre after an animal's time budget


class GuidanceFSM:
    def __init__(self, cfg: FSMConfig | None = None):
        self.cfg = cfg or FSMConfig()
        self.state = State.SEARCH
        self.since = 0.0
        self.recent: deque[bool] = deque(maxlen=self.cfg.acquire_m)
        self.recording = False
        self.rec_since = 0.0
        self.film_ok_since: float | None = None
        self.left_active_at: float | None = None
        self.end_reason: str | None = None

    def _go(self, s: State, t: float) -> None:
        self.state, self.since = s, t

    def update(
        self,
        t: float,
        frame_prob: float,
        target: Target,
        track: TrackState,
        release: str | bool | None = None,
    ) -> list[str]:
        """Advance one frame. ``release``: leave the current animal now - a reason such as
        ``"enough"``, ``"fled"``, ``"no_shot"`` or ``"budget"`` (``True`` means ``"budget"``)."""
        c = self.cfg
        events: list[str] = []
        prev = self.state
        detected = (frame_prob >= c.frame_on or target.peak >= c.heat_on) and target.found
        self.recent.append(detected)
        if self.state in ENGAGED and release:
            self._go(State.RELEASE, t)
            self.end_reason = release if isinstance(release, str) else "budget"
            self.film_ok_since = None
            events.append("encounter_end")
        elif self.state == State.SEARCH:
            if sum(self.recent) >= c.acquire_n:
                self._go(State.ACQUIRE, t)
        elif self.state == State.ACQUIRE:
            if track.confirmed:
                self._go(State.TRACK, t)
                self.end_reason = None
                events.append("encounter_start")
            elif t - self.since > c.acquire_timeout_s or not track.active:
                self._go(State.SEARCH, t)
                self.recent.clear()  # re-acquiring needs fresh evidence (no SEARCH/ACQUIRE flapping)
        elif self.state in (State.TRACK, State.FILM):
            if not track.active or track.misses > 2:
                self._go(State.LOST, t)
            else:
                centred = abs(track.yaw) < c.film_center_deg and abs(track.pitch) < c.film_center_deg
                good_size = c.film_size[0] <= track.size <= c.film_size[1]
                if centred and good_size:
                    self.film_ok_since = self.film_ok_since if self.film_ok_since is not None else t
                    if self.state == State.TRACK and t - self.film_ok_since >= c.film_hold_s:
                        self._go(State.FILM, t)
                else:
                    self.film_ok_since = None
                    if self.state == State.FILM:
                        self._go(State.TRACK, t)
        elif self.state == State.LOST:
            if track.active and track.confirmed and track.misses == 0:
                self._go(State.TRACK, t)
            elif t - self.since > c.lost_timeout_s:
                self._go(State.SEARCH, t)
                self.recent.clear()
                self.end_reason = "lost"
                events.append("encounter_end")
        elif self.state == State.RELEASE:
            if t - self.since >= c.release_s:
                self._go(State.SEARCH, t)
                self.recent.clear()

        if self.state in ENGAGED:
            self.left_active_at = None
            if not self.recording:
                self.recording, self.rec_since = True, t
                events.append("start_recording")
            elif t - self.rec_since > c.max_record_s:
                events += ["stop_recording", "start_recording"]  # roll over into a new file
                self.rec_since = t
        elif self.recording:
            self.left_active_at = self.left_active_at if self.left_active_at is not None else t
            if t - self.left_active_at >= c.postroll_s:
                self.recording = False
                events.append("stop_recording")
        if self.state != prev:
            events.append(f"state:{self.state.value}")
        return events
