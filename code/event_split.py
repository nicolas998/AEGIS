import numpy as np
import pandas as pd
from typing import List, Tuple
from scipy.signal import find_peaks


def split_events(
    q: pd.Series,
    runoff_mask: np.ndarray,
    missing: np.ndarray,
    event_threshold: float = 0.2,
    min_rise: float = 0.1,
    min_peakness: float = 0.2,
    max_nans: float = 0.2,
    smooth_hours: int = 0,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Peak-based event separation on top of the EventDetector runoff windows.

    EventDetector keeps one event per start-to-next-start runoff window, so
    storms whose runoff does not return to zero in between are merged and
    only the largest peak survives. Here every window is split at the trough
    between two peaks above `event_threshold` when both peaks rise at least
    `min_rise` above that trough. Each sub-event starts at the lowest point
    before its peak and ends at the next trough (or the window end).

    Args:
        q: normalized hourly discharge (gaps interpolated), datetime index
        runoff_mask: binary runoff series from EventDetector (1 = runoff > 0)
        missing: boolean array, True where the original data were missing
        event_threshold: minimum normalized peak of an event
        min_rise: minimum rise of a peak above its separating trough/start
        min_peakness: (peak - start) / peak must exceed this
        max_nans: max fraction of missing data allowed in an event
        smooth_hours: centered moving-average window used to locate peaks and
            troughs (e.g. 25 h removes tidal oscillations); 0 disables it

    Returns:
        list of (start, end, peak_time) tuples
    """
    raw = q.values
    v = (q.rolling(smooth_hours, center=True, min_periods=1).mean().values
         if smooth_hours > 1 else raw)
    starts = np.where(np.diff(runoff_mask) == 1)[0]
    events = []
    for a, b in zip(starts[:-1], starts[1:]):
        seg = v[a:b + 1]
        if seg.max() <= event_threshold:
            continue

        # merge candidate peaks left to right; cut where the trough is deep enough
        cand = list(find_peaks(seg, height=event_threshold)[0]) or [int(seg.argmax())]
        cuts, cur = [], cand[0]
        for nxt in cand[1:]:
            trough = cur + int(np.argmin(seg[cur:nxt + 1]))
            if min(seg[cur], seg[nxt]) - seg[trough] >= min_rise:
                cuts.append(trough)
                cur = nxt
            elif seg[nxt] > seg[cur]:
                cur = nxt

        bounds = [0] + cuts + [len(seg) - 1]
        for s, e in zip(bounds[:-1], bounds[1:]):
            sub = seg[s:e + 1]
            p = int(sub.argmax())
            s0 = int(np.argmin(sub[:p + 1]))       # start of the rising limb
            rise = sub[p] - sub[s0]
            if (sub[p] > event_threshold and rise >= min_rise
                    and rise / sub[p] > min_peakness
                    and missing[a + s + s0:a + e + 1].mean() < max_nans):
                lo, hi = a + s + s0, a + e
                peak = lo + int(np.argmax(raw[lo:hi + 1]))
                events.append((q.index[lo], q.index[hi], q.index[peak]))
    return events
