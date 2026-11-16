import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('sequence_recovery_audit',Path(__file__).resolve().parents[1]/'scripts/audit_sequence_recovery.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def recovery_result():
    impact=dict(direction_error_deg=1.,preferred_foot=True,relative_forward_speed_error=.1)
    return dict(requested=1,initial_skill='pass',handoff_seed=None,
        pass_impacts={'0':dict(impact,tick=5,target_speed_mps=2.5)},
        shoot_impacts={'0':dict(impact,tick=20,target_speed_mps=6.)},
        first_resets=[],switches=[dict(tick=15,environments=[0],robot_and_history_preserved=True)],
        switched=1,recovery_steps=10,post_shoot_recovery_steps=10,post_shoot_recovery_speed=2.5,
        recovery_switches=[dict(tick=30,environments=[0],robot_and_history_preserved=True,
                               task_target_speed=[6.],planner_target_speed=[2.5])],
        recovery_started=1,both_observed_kicks=1,both_direction_le_10deg=1,
        both_direction_and_preferred_foot=1,both_direction_preferred_foot_and_speed_30pct=1,
        both_accurate_and_continuous_through_end=1,both_kicks_and_no_reset_through_end=1,
        continuous_survivors=1,both_direction_foot_speed_and_continuous_through_end=1)


def test_recovery_override_preserves_full_success_accounting():
    result=module.validate(recovery_result())
    assert isinstance(result['derived_counts'],dict)
    assert result['derived_counts']['both_direction_foot_speed_and_continuous_through_end']==1


@pytest.mark.parametrize('field,value',[('task_target_speed',2.5),('planner_target_speed',6.)])
def test_recovery_rejects_changed_scoring_or_missing_override(field,value):
    result=recovery_result()
    result['recovery_switches'][0][field]=[value]
    with pytest.raises(AssertionError):module.validate(result)
