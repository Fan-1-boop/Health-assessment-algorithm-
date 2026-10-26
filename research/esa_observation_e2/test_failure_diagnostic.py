"""Synthetic NUMERICAL fixtures only; never interpreted as satellite research data."""
import numpy as np
from research.esa_observation_e2.failure_diagnostic import event_audit
from v13_protocol import Partition


def record(ident, start, stop):
    return {
        "id": ident, "target_evaluable": True,
        "global_start": start, "global_stop": stop,
        "target_start": start, "target_stop": stop,
    }


def test_missed_event_single_crossing_is_not_a_sustained_alarm():
    ends = np.arange(0, 80, 4)
    codes = np.zeros(100, dtype=int)
    codes[16:40] = 1
    scores = np.zeros(len(ends))
    scores[np.flatnonzero(ends == 24)] = 1.0
    events, alarms = event_audit(
        [record("a", 16, 40)], Partition("test", 0, 100, ()),
        ends, codes, scores, 0.8
    )
    assert len(events) == 1
    assert events[0]["observed_failure_type"] == "threshold_crossings_without_in_event_sustained_alarm"
    assert events[0]["instantaneous_validation_threshold_crossings"] == 1
    assert events[0]["max_consecutive_in_event_crossings"] == 1
    assert not alarms.any()


def test_three_consecutive_crossings_produce_new_onset():
    ends = np.arange(0, 80, 4)
    codes = np.zeros(100, dtype=int)
    codes[16:40] = 1
    scores = np.zeros(len(ends))
    scores[np.isin(ends, [16, 20, 24])] = 1.0
    events, alarms = event_audit(
        [record("a", 16, 40)], Partition("test", 0, 100, ()),
        ends, codes, scores, .8
    )
    assert events[0]["observed_failure_type"] == "in_event_new_onset"
    assert events[0]["max_consecutive_in_event_crossings"] == 3
    assert events[0]["strict_alarm_delay_10min_grid_steps"] == 8
    assert alarms[np.flatnonzero(ends == 24)[0]]


def test_preexisting_alarm_does_not_count_as_new_onset():
    ends = np.arange(0, 80, 4)
    codes = np.zeros(100, dtype=int)
    codes[16:40] = 1
    scores = np.zeros(len(ends))
    scores[np.isin(ends, [4, 8, 12, 16, 20, 24])] = 1.0
    events, _ = event_audit(
        [record("a", 16, 40)], Partition("test", 0, 100, ()),
        ends, codes, scores, .8
    )
    assert events[0]["observed_failure_type"] == "alarm_preexisting"
    assert not events[0]["strict_new_alarm_onset"]


def test_no_signal_and_gap_breaks_sustained_runs():
    ends = np.array([4, 8, 20, 24, 28])
    codes = np.zeros(100, dtype=int)
    codes[4:40] = 1
    e = [record("a", 4, 40)]
    out, alarms = event_audit(
        e, Partition("test", 0, 100, ()), ends, codes,
        np.ones(len(ends)), .8
    )
    assert not alarms[2] and alarms[-1]
    out2, alarm2 = event_audit(
        e, Partition("test", 0, 100, ()), ends, codes,
        np.zeros(len(ends)), .8
    )
    assert not alarm2.any()
    assert out2[0]["observed_failure_type"] == "no_in_event_single_threshold_crossing"
