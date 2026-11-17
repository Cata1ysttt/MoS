"""Wait for the live zero-control screen, then run one finite training smoke."""
import json
import os
import subprocess
import time
from run_target_guided_planner import ROOT,read,sha,container

TAG='planner_post_pass_latent_ppo_smoke_r1'

def main():
    runs=ROOT/'runs';status=runs/f'{TAG}_queue_status.json';assert not status.exists()
    zero=runs/'planner_post_pass_latent_adapter_zero_r1_screen_status.json'
    checkpoint=runs/'planner_post_pass_latent_adapter_zero_r1/diagnostic.pt'
    output=runs/TAG;assert not output.exists()
    checksum=sha(checkpoint);assert checksum=='81dc1618e1871913e32decde4611993bda56423a3c723bee82ace4ffa393233b'
    def state(stage,**details):
        temp=status.with_suffix('.tmp');temp.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),updated=time.time(),**details),indent=2)+'\n');temp.replace(status)
    try:
        while True:
            screen=read(zero)
            if screen['stage'] in ('completed','failed'):break
            os.kill(screen['pid'],0)
            state('waiting_for_zero_screen',dependency=str(zero),dependency_pid=screen['pid']);time.sleep(10)
        assert screen['stage']=='completed','Zero adapter behavior preservation failed'
        assert screen['results']['sequence']['summary_preserved_except_declared_fields']
        assert screen['results']['sequence']['adapter_audit']['active_control_steps']>0
        sources=dict(screen['results']['protocol']['runtime_source_sha256'])
        for name in ['scripts/train_post_pass_latent_ppo.py','scripts/queue_post_pass_latent_ppo_smoke.py']:
            sources[name]=sha(ROOT/name)
        assert all(sha(ROOT/path)==digest for path,digest in sources.items())
        protocol=dict(checkpoint_sha256=checksum,runtime_source_sha256=sources,
            seed=817100,num_envs=32,iterations=16,rollout_steps=32,horizon_seconds=8,
            smoke_only=True,automatic_full_training=False,automatic_promotion=False)
        (runs/f'{TAG}_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        state('training_smoke',protocol=protocol)
        command=['/workspace/fiber-soccer/scripts/train_post_pass_latent_ppo.py','--headless','--device','cuda:0',
            '--checkpoint',container(checkpoint),'--output',container(output),'--seed','817100',
            '--num-envs','32','--iterations','16','--rollout-steps','32','--smoke-test']
        with output.with_suffix('.log').open('x') as log:
            subprocess.run([str(ROOT/'scripts/container_exec.sh'),*command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,MOS_CUDA_VISIBLE_DEVICES='3'),check=True)
        result=read(output/'complete.json')
        assert result['smoke_test'] and result['deployable_roundtrip'] and result['shared_tensors_unchanged']
        assert result['active_policy_steps']>0 and result['actor_updates']>0 and result['timeouts']>0
        assert result['one_stateful_call_per_observation'] and result['minibatch_updates_stateless']
        assert result['baseline_calls']==513 and result['environment_steps']==16384
        assert sha(checkpoint)==checksum and sha(output/'final.pt')==result['checkpoint_sha256']
        assert all(sha(ROOT/path)==digest for path,digest in sources.items())
        state('completed',protocol=protocol,result=result)
    except Exception as error:
        state('failed',error=repr(error));raise

if __name__=='__main__':main()
