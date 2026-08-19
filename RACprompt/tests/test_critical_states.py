import numpy as np

from racprompt.critical_states import select_critical_states


def test_selector_returns_unique_target_and_temporal_coverage():
    rng = np.random.default_rng(2)
    dplus = rng.random(512)
    compatibility = rng.random(512)
    selected = select_critical_states(dplus, compatibility)
    positions = [state.position for state in selected]
    assert len(selected) == 24
    assert len(set(positions)) == len(positions)
    assert all(0 <= position < 512 for position in positions)
    assert sum("segment" in state.reasons for state in selected) == 12


def test_short_trajectory_uses_every_valid_position():
    selected = select_critical_states(np.arange(7.0), np.zeros(7), target=24)
    assert [state.position for state in selected] == list(range(7))


def test_global_peak_uses_dplus_not_ta_opd_product():
    dplus = np.array([0.1, 0.8, 0.2, 1.0])
    compatibility = np.array([1.0, 1.0, 1.0, 0.0])
    selected = select_critical_states(
        dplus,
        compatibility,
        target=1,
        num_segments=1,
        global_peaks=1,
        change_points=0,
        min_gap_tokens=0,
    )
    # TA-OPD D*C would reject position 3, whereas RAC correction magnitude selects it.
    assert selected[0].position == 3


def test_selector_records_valid_reason_names():
    states = select_critical_states(np.linspace(0, 1, 100), np.sin(np.arange(100)))
    allowed = {"segment", "global_peak", "change_point", "fill"}
    assert all(set(state.reasons) <= allowed for state in states)
