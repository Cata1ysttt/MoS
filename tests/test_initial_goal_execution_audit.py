"""Recovery execution must match its declared scope and real live intervals."""
import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('recovery_execution_audit',Path(__file__).parents[1]/'scripts/audit_initial_goal_execution.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


@pytest.mark.parametrize('scope',['initial_pass','initial_pass_and_post_pass_shoot'])
def test_recovery_audit_checks_scope_and_failure_inclusive_interval(tmp_path,scope):
    summary=dict(standing_recovery_mode='initial_backward_goal',predicted_target_refinement_scope=scope,
        steps=4,requested=1,first_resets=[dict(environment=0,tick=2)],
        recovery_switches=[dict(tick=1,environments=[0])],standing_recovery_active_control_steps=2)
    records=[dict(standing_recovery_mode='initial_backward_goal',refinement_scope=scope,
        standing_recovery_environments=value) for value in [0,1,1,0]]
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    (tmp_path/'initial_backward_goal.pt').write_bytes(b'audit fixture')
    record_path=tmp_path/'predicted_target_solver.json';record_path.write_text(json.dumps(records))
    result=module.audit(tmp_path)
    assert result['active_control_steps']==2 and result['alive_recovery_counts_exact']
    records[0]['refinement_scope']='unexpected';record_path.write_text(json.dumps(records))
    with pytest.raises(AssertionError):module.audit(tmp_path)
    records[0]['refinement_scope']=scope
    records[2]['standing_recovery_environments']=0;record_path.write_text(json.dumps(records))
    with pytest.raises(AssertionError,match='Recovery counts'):module.audit(tmp_path)
