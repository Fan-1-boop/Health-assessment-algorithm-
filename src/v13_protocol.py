"""V1.3 pre-registered event-first, no-overlap evaluation protocol.

ESA-ADB official preprocessed `is_anomaly_<channel>` codes:
0 nominal, 1 anomaly, 2 rare nominal, 3 gap, 4 invalid.
Do NOT merge 1 and 2 for the strict anomaly-vs-rare-nominal study.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

ESA_CODES = {'nominal': 0, 'anomaly': 1, 'rare_nominal': 2, 'gap': 3, 'invalid': 4}


@dataclass(frozen=True)
class Event:
    start: int
    stop: int  # exclusive
    label: int  # 1 or 2


@dataclass(frozen=True)
class Partition:
    name: str
    start: int
    stop: int
    events: tuple[Event, ...]


def read_esa_preprocessed(path: str | Path, channel: str, max_rows: int | None = None):
    """Read an official Mission1 *preprocessed* CSV, not a raw Zip archive.

    Returns [n, d] raw measured signals and [n] official codes for target
    channel. All is_anomaly_* annotation columns are excluded from model input.
    """
    path = Path(path)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    label_col = f'is_anomaly_{channel}'
    if 'timestamp' not in header or channel not in header or label_col not in header:
        raise ValueError(f'Expected timestamp, {channel}, {label_col} from official ESA preprocessor')
    # No future target/label features; additional channels/telecommands allowed.
    signals = [c for c in header if c != 'timestamp' and not c.startswith('is_anomaly_')]
    if not signals:
        raise ValueError('No raw channels found')
    df = pd.read_csv(path, usecols=['timestamp', *signals, label_col], nrows=max_rows)
    if df['timestamp'].duplicated().any() or not df['timestamp'].is_monotonic_increasing:
        raise ValueError('Unsorted or duplicated timestamps; preserve official chronological order')
    codes = df[label_col].to_numpy(dtype=int)
    if not np.isin(codes, np.arange(5)).all():
        raise ValueError('Unexpected ESA annotation code')
    X = df[signals].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    if not np.isfinite(X).all():
        # Missingness must not be imputed across validation/test boundaries.
        raise ValueError('Nonfinite signal value; clean within partitions without future information')
    return X, codes, df['timestamp'].to_numpy(), signals


def find_events(codes):
    """Find maximal contiguous runs of event type 1 or 2, separated by 0/3/4.

    Interleaved anomaly+rare codes split at their different type; the official
    ID-based annotations should be preferred when two events overlap.
    """
    c = np.asarray(codes, dtype=int)
    events = []
    i = 0
    while i < len(c):
        if c[i] not in (1, 2):
            i += 1
            continue
        label, start = int(c[i]), i
        i += 1
        while i < len(c) and c[i] == label:
            i += 1
        events.append(Event(start, i, label))
    return events


def event_first_partition(codes, min_anomalies_per_split=2):
    """Chronologically split *events* before constructing any windows.

    Every timestamp belongs to one non-overlapping partition. Partition edges
    are midpoints between anomaly-event groups, not individual windows.
    Rare-normal events remain in their chronological partition and are not
    mislabeled as anomalies. Insufficient events --> error, not window shuffling.
    """
    codes = np.asarray(codes, int)
    events = find_events(codes)
    anomalies = [e for e in events if e.label == 1]
    n = len(anomalies)
    if n < 3 * min_anomalies_per_split:
        raise ValueError(f'Insufficient independent anomaly episodes: {n}; require at least {3*min_anomalies_per_split}')
    k1 = max(min_anomalies_per_split, int(np.floor(n * .6)))
    k2 = max(k1 + min_anomalies_per_split, int(np.floor(n * .8)))
    if n-k2 < min_anomalies_per_split:
        k2 = n-min_anomalies_per_split
    if k2-k1 < min_anomalies_per_split:
        k1 = k2-min_anomalies_per_split
    if k1 < min_anomalies_per_split:
        raise ValueError('No valid three-way chronological event split')
    def cut(prev, nxt):
        if prev.stop > nxt.start:
            raise ValueError('Overlapping anomaly episodes: use unique official IDs')
        return (prev.stop + nxt.start) // 2
    b1 = cut(anomalies[k1-1], anomalies[k1])
    b2 = cut(anomalies[k2-1], anomalies[k2])
    if not (0 < b1 < b2 < len(codes)):
        raise ValueError('Event partition boundaries invalid')
    bounds = [(0, b1), (b1, b2), (b2, len(codes))]
    partitions = tuple(Partition(name, a, b, tuple(e for e in events if a <= e.start < b))
                       for name, (a,b) in zip(('train','validation','test'), bounds))
    for part in partitions:
        if sum(e.label == 1 for e in part.events) < min_anomalies_per_split:
            raise ValueError(f'{part.name} lacks sufficient anomaly episodes')
        if any(e.stop > part.stop for e in part.events):
            raise ValueError('An event would cross a partition boundary')
    return partitions


def windows_in_partition(raw, codes, part, lookback=64, stride=1):
    """Causal windows fully inside one partition; invalid/gaps excluded.

    Training/test windows cannot share any source observation because the
    partition intervals are disjoint even before window construction.
    """
    raw = np.asarray(raw, float); c = np.asarray(codes, int)
    if raw.ndim == 1:
        raw = raw[:,None]
    if len(raw) != len(c) or lookback < 2 or stride < 1:
        raise ValueError('Invalid shape, lookback, or stride')
    ends = np.arange(part.start + lookback - 1, part.stop, stride, dtype=int)
    if not len(ends):
        raise ValueError(f'{part.name}: no causal windows')
    bad = (c >= 3).astype(int)
    cumulative = np.r_[0, bad.cumsum()]
    good = cumulative[ends+1] == cumulative[ends+1-lookback]
    ends = ends[good]
    if not len(ends):
        raise ValueError(f'{part.name}: all windows invalid')
    take = ends[:,None] - np.arange(lookback-1, -1, -1)[None,:]
    assert int(take.min()) >= part.start and int(take.max()) < part.stop
    return raw[take], ends, (c[ends] == 1).astype(int), (c[ends] == 2).astype(int)


def diagnostic_manifest(parts, codes):
    return {'label_semantics': '0 nominal / 1 anomaly / 2 rare nominal / 3 gap / 4 invalid',
            'splits': {p.name: {'sample_start': p.start, 'sample_stop_exclusive': p.stop,
                    'anomaly_events': sum(e.label == 1 for e in p.events),
                    'rare_nominal_events': sum(e.label == 2 for e in p.events)} for p in parts}}
