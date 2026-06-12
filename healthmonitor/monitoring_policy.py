"""Shared formal monitoring policy used by offline replay and the dashboard."""
from dataclasses import asdict, dataclass
from typing import Iterable, Optional

import numpy as np


POLICY_CONFIG_VERSION = "overall-risk-v1"


@dataclass(frozen=True)
class AlarmPolicyConfig:
    score: str = "1-P(Normal)"
    ewma_alpha: float = 0.65
    threshold: float = 0.10
    consecutive_k: int = 2
    inference_interval_sec: float = 10.0
    version: str = POLICY_CONFIG_VERSION


@dataclass(frozen=True)
class AlarmPolicyState:
    ewma_risk: Optional[float] = None
    consecutive_count: int = 0
    alert_active: bool = False
    status: str = "normal"


DEFAULT_POLICY = AlarmPolicyConfig()


def policy_config_dict(config: AlarmPolicyConfig = DEFAULT_POLICY):
    return asdict(config)


def update_alarm_policy(
    state: AlarmPolicyState,
    overall_risk_raw: float,
    config: AlarmPolicyConfig = DEFAULT_POLICY,
):
    """Advance the causal EWMA + consecutive-trigger policy by one inference."""
    risk = float(np.clip(overall_risk_raw, 0.0, 1.0))
    if state.ewma_risk is None:
        ewma = risk
    else:
        ewma = (
            config.ewma_alpha * risk
            + (1.0 - config.ewma_alpha) * float(state.ewma_risk)
        )

    above = ewma >= config.threshold
    consecutive = state.consecutive_count + 1 if above else 0
    alert_active = consecutive >= config.consecutive_k
    if alert_active:
        status = "alert"
    elif above:
        status = "fluctuation"
    else:
        status = "normal"

    next_state = AlarmPolicyState(
        ewma_risk=float(ewma),
        consecutive_count=int(consecutive),
        alert_active=bool(alert_active),
        status=status,
    )
    transition = None
    if not state.alert_active and next_state.alert_active:
        transition = "alarm_started"
    elif state.alert_active and not next_state.alert_active:
        transition = "alarm_cleared"
    elif state.alert_active and next_state.alert_active:
        transition = "alarm_continues"
    return next_state, transition


def apply_alarm_policy(
    risks: Iterable[float],
    config: AlarmPolicyConfig = DEFAULT_POLICY,
):
    """Apply the exact serving policy to an ordered risk sequence."""
    states = []
    transitions = []
    state = AlarmPolicyState()
    for risk in risks:
        state, transition = update_alarm_policy(state, risk, config)
        states.append(state)
        transitions.append(transition)
    return states, transitions


def apply_alarm_policy_by_group(
    risks,
    groups,
    config: AlarmPolicyConfig = DEFAULT_POLICY,
):
    """Apply the policy independently to contiguous patient/record groups."""
    risks = np.asarray(risks, dtype=float)
    groups = np.asarray(groups)
    if len(risks) != len(groups):
        raise ValueError("risks and groups must have the same length")

    ewma = np.zeros(len(risks), dtype=float)
    alert = np.zeros(len(risks), dtype=bool)
    status = np.empty(len(risks), dtype=object)
    previous_group = object()
    state = AlarmPolicyState()
    for idx, (risk, group) in enumerate(zip(risks, groups)):
        if idx == 0 or group != previous_group:
            state = AlarmPolicyState()
        state, _ = update_alarm_policy(state, risk, config)
        ewma[idx] = float(state.ewma_risk)
        alert[idx] = state.alert_active
        status[idx] = state.status
        previous_group = group
    return ewma, alert, status
