"""Verify gate equivalence with the same zero controller in already-failed lanes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from audit_sequence_recovery import validate
root=Path(__file__).resolve().parents[1]
status=root/'runs/planner_post_pass_gate_zero_control_status.json';assert not status.exists()
source=root/'runs/planner_sequence_shoot_adapter_r1/final.pt'
gated=root/'runs/planner_sequence_shoot_adapter_r1_post_pass/final.pt'
results={}
def state(stage,**details):
    status.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),updated=time.time(),results=results,**details),indent=2))
def container(path):return '/workspace/fiber-soccer/'+str(path.relative_to(root))
try:
    original_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,gated]}
    protocols=[];contracts=[]
    for label,checkpoint in [('source',source),('gated',gated)]:
        destination=root/'runs'/f'planner_post_pass_gate_zero_control_{label}'
        state('evaluating_'+label)
        cmd=[str(root/'scripts/container_exec.sh'),'/workspace/fiber-soccer/scripts/probe_soccer_sequence.py',
            '--headless','--device','cuda:0','--checkpoint',container(checkpoint),'--output',container(destination),
            '--num-envs','32','--steps','1000','--seed','815301','--handoff-seed','815301',
            '--scene-base','shoot','--action-contract','shoot','--reset-contract','shoot','--events-contract','shoot',
            '--incoming-pass-command','--continuous-after-kick','--transition-steps','0','--post-shoot-recovery-steps','10',
            '--failed-lane-control','zero']
        if label=='gated':cmd+=['--audit-post-pass-gate']
        with destination.with_suffix('.log').open('x') as f:
            subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,MOS_CUDA_VISIBLE_DEVICES='3'),check=True)
        summary=json.loads((destination/'summary.json').read_text())
        assert summary['failed_lane_control']=='zero' and not summary['expert_loaded'] and summary['one_policy_instance']
        results[label]=dict(summary=summary,accounting=validate(summary))
        protocols.append(json.loads((destination/'protocol.json').read_text()))
        contracts.append(json.loads((destination/'scene_contracts.json').read_text()))
        if label=='gated':
            results['gate_runtime_audit']=json.loads((destination/'gate_runtime_audit.json').read_text())
    a,b=[results[k]['summary'] for k in ['source','gated']]
    ignored={'checkpoint','checkpoint_sha256','shoot_adapter_after_pass_only'}
    assert {k:v for k,v in a.items() if k not in ignored}=={k:v for k,v in b.items() if k not in ignored}
    assert contracts[0]==contracts[1]
    assert {k:v for k,v in protocols[0].items() if k!='shoot_adapter_after_pass_only'}=={k:v for k,v in protocols[1].items() if k!='shoot_adapter_after_pass_only'}
    audit=results['gate_runtime_audit']
    assert audit['alive_states_checked']>0 and audit['alive_latent_outputs_exact'] and audit['alive_joint_targets_exact']
    assert audit['comparison_rng_preserved'] and audit['failed_states_with_different_latents']>0
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest for p,digest in original_hashes.items())
    state('completed',entire_trajectory_summary_equal=True,checkpoint_hashes=original_hashes,
        scope='Controlled replay on a reused seed. Both models apply identical raw zero actions after first failure; failed lanes remain failed. Tests routing and simulation-protocol equivalence, not improved task performance.')
except Exception as error:
    state('failed',error=repr(error));raise
