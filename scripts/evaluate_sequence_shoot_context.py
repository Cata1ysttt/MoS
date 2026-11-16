"""Diagnose the shooting teacher after passing, and verify post-pass deployment gating."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from audit_sequence_recovery import validate

root=Path(__file__).resolve().parents[1]
status=root/'runs/planner_sequence_shoot_context_status.json';assert not status.exists()
upstream=root/'runs/planner_sequence_shoot_adapter_r1_status.json'
base=root/'runs/planner_history_r3_contact_train/best.pt'
source=root/'runs/planner_sequence_shoot_adapter_r1/final.pt'
gated=root/'runs/planner_sequence_shoot_adapter_r1_post_pass/final.pt'
results={}
def container(path):return '/workspace/fiber-soccer/'+str(path.relative_to(root))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def state(stage,**details):
    status.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),updated=time.time(),results=results,**details),indent=2))
def execute(cmd,destination):
    with destination.with_suffix('.log').open('x') as f:
        subprocess.run([str(root/'scripts/container_exec.sh')]+cmd,stdout=f,stderr=subprocess.STDOUT,
            env=dict(os.environ,MOS_CUDA_VISIBLE_DEVICES='3'),check=True)
def sequence(model,destination,expert=False):
    cmd=['/workspace/fiber-soccer/scripts/probe_soccer_sequence.py','--headless','--device','cuda:0',
        '--checkpoint',container(model),'--output',container(destination),'--num-envs','32','--steps','1000',
        '--seed','815301','--handoff-seed','815301','--scene-base','shoot','--action-contract','shoot',
        '--reset-contract','shoot','--events-contract','shoot','--incoming-pass-command',
        '--continuous-after-kick','--transition-steps','0','--post-shoot-recovery-steps','10']
    if expert:cmd+=['--shoot-expert-diagnostic']
    execute(cmd,destination)
    return json.loads((destination/'summary.json').read_text())
try:
    while True:
        u=json.loads(upstream.read_text())
        if u['stage']=='completed':break
        if u['stage']=='failed':raise RuntimeError(u)
        process=Path(f"/proc/{u['pid']}/cmdline")
        assert process.exists() and b'run_sequence_shoot_adapter.py' in process.read_bytes()
        state('waiting_for_candidate_evaluation',upstream_pid=u['pid']);time.sleep(10)
    original_hashes={str(p):sha(p) for p in [base,source,gated]}
    state('evaluating_teacher_after_pass')
    destination=root/'runs/planner_sequence_shoot_teacher_context'
    teacher=sequence(base,destination,expert=True)
    reference=root/'runs/planner_sequence_shoot_adapter_r1_815301_baseline'
    baseline=json.loads((reference/'summary.json').read_text())
    assert teacher['expert_loaded'] and not teacher['one_policy_instance']
    assert teacher['initial_state_sha256']==baseline['initial_state_sha256']
    assert json.loads((destination/'scene_contracts.json').read_text())==json.loads((reference/'scene_contracts.json').read_text())
    protocols=[json.loads((p/'protocol.json').read_text()) for p in [destination,reference]]
    assert {k:v for k,v in protocols[0].items() if k!='shoot_controller'}=={k:v for k,v in protocols[1].items() if k!='shoot_controller'}
    first=min(row['tick'] for row in baseline['switches'])
    for key in ['pass_impacts','shoot_impacts']:
        assert {k:v for k,v in teacher[key].items() if v['tick']<=first}=={k:v for k,v in baseline[key].items() if v['tick']<=first},key
    for key in ['switches','first_resets','contact_observations']:
        assert [v for v in teacher[key] if v['tick']<=first]==[v for v in baseline[key] if v['tick']<=first],key
    results['teacher_diagnostic']=dict(summary=teacher,accounting=validate(teacher),
        initial_state_and_pre_handoff_prefix_equal=True,first_handoff_tick=first,
        scope='Original ONNX actions during shooting only; shared r3 passing before handoff and after shooting recovery switch. Not a unified-policy success result.')
    state('verifying_standalone_shoot_gate')
    destination=root/'runs/planner_sequence_shoot_adapter_r1_post_pass_shoot_eval'
    execute(['/workspace/fiber-soccer/scripts/probe_shoot_expert.py','--headless','--device','cuda:0',
        '--evaluate','--seed','811314','--num-envs','16','--steps','2000',
        '--projector-checkpoint',container(gated),'--output',container(destination)],destination)
    s=json.loads((destination/'summary.json').read_text())
    reference=root/'runs/planner_history_r3_contact_shoot_eval'
    assert s['shoot_adapter_after_pass_only'] and s['completed']==s['requested']==16
    assert s['initial_state_sha256']==json.loads((reference/'summary.json').read_text())['initial_state_sha256']
    for name in ['episodes.json','first_kick_velocity.json']:
        assert json.loads((destination/name).read_text())==json.loads((reference/name).read_text()),name
    results['gated_native_shoot']=dict(summary=s,episode_and_impact_records_equal_to_r3=True)
    state('verifying_continuous_gate_equivalence')
    destination=root/'runs/planner_sequence_shoot_adapter_r1_post_pass_sequence'
    controlled=sequence(gated,destination)
    reference=root/'runs/planner_sequence_shoot_adapter_r1_815301_candidate'
    original=json.loads((reference/'summary.json').read_text())
    assert controlled['shoot_adapter_after_pass_only'] and not controlled['expert_loaded']
    ignored={'checkpoint','checkpoint_sha256','shoot_adapter_after_pass_only'}
    assert {k:v for k,v in controlled.items() if k not in ignored}=={k:v for k,v in original.items() if k not in ignored}
    for name in ['protocol.json','scene_contracts.json']:
        left=json.loads((destination/name).read_text());right=json.loads((reference/name).read_text())
        left.pop('shoot_adapter_after_pass_only',None)
        assert left==right,name
    results['gated_continuous']=dict(summary=controlled,accounting=validate(controlled),entire_trajectory_summary_equal=True)
    assert all(sha(Path(p))==digest for p,digest in original_hashes.items())
    state('completed',checkpoint_hashes=original_hashes,
        scope='Teacher capability diagnosis plus deployment-context verification, not new independent validation or automatic promotion.')
except Exception as error:
    state('failed',error=repr(error));raise
