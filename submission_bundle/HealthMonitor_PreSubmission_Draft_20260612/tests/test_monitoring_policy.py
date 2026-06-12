"""Unit tests for the shared formal causal alarm policy."""
import numpy as np

from healthmonitor.monitoring_policy import (
    AlarmPolicyState,
    DEFAULT_POLICY,
    apply_alarm_policy,
    apply_alarm_policy_by_group,
    update_alarm_policy,
)


def main_test():
    state = AlarmPolicyState()
    state, transition = update_alarm_policy(state, 0.20)
    assert state.status == "fluctuation"
    assert not state.alert_active
    assert transition is None

    state, transition = update_alarm_policy(state, 0.20)
    assert state.status == "alert"
    assert state.alert_active
    assert transition == "alarm_started"

    state, transition = update_alarm_policy(state, 0.0)
    assert state.status == "normal"
    assert not state.alert_active
    assert transition == "alarm_cleared"

    risks = np.array([0.0, 0.2, 0.2, 0.0, 0.2, 0.2])
    states, transitions = apply_alarm_policy(risks)
    assert [s.alert_active for s in states] == [
        False, False, True, False, False, True
    ]
    assert transitions[2] == "alarm_started"
    assert transitions[3] == "alarm_cleared"

    ewma, alert, status = apply_alarm_policy_by_group(
        [0.2, 0.2, 0.2, 0.2],
        ["a", "a", "b", "b"],
        DEFAULT_POLICY,
    )
    assert np.allclose(ewma, 0.2)
    assert alert.tolist() == [False, True, False, True]
    assert status.tolist() == ["fluctuation", "alert", "fluctuation", "alert"]
    print("Monitoring policy tests passed.")


if __name__ == "__main__":
    main_test()
