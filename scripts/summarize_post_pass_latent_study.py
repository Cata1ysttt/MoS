"""Recompute completed latent-PPO study results from the original evaluation files."""
import argparse
import hashlib
import json
from pathlib import Path
from summarize_continuous_shoot_ppo import outcomes

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag',choices=['planner_post_pass_latent_ppo_r1','planner_post_pass_latent_ppo_budget_r1','planner_post_pass_latent_ppo_low_noise_r1'],default='planner_post_pass_latent_ppo_budget_r1')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-historical-source-changes',action='store_true',help='Only for validating the already archived 100-iteration study; list later source changes explicitly')
    args=parser.parse_args();assert not args.output.exists()
    assert not args.allow_historical_source_changes or args.tag=='planner_post_pass_latent_ppo_r1'
    runs=ROOT/'runs';run=runs/args.tag;state=read(runs/f'{args.tag}_status.json')
    assert state['stage']=='completed','Wait for training, native regression and both paired evaluations'
    results=state['results'];protocol=results['protocol']
    changed=[name for name,digest in protocol['runtime_source_sha256'].items() if sha(ROOT/name)!=digest]
    assert not changed or args.allow_historical_source_changes,'Frozen runtime files changed'
    assert sha(ROOT/'AH-BFM/AH-BFM.pt')==protocol['actor_sha256']
    assert sha(runs/'planner_post_pass_latent_adapter_zero_r1/diagnostic.pt')==protocol['source_sha256']
    assert sha(runs/'planner_new_ball_task_history_r1/diagnostic.pt')==protocol['control_sha256']
    training=read(run/'complete.json');audit=read(runs/f'{args.tag}_audit.json');config=read(run/'config.json')
    assert training==results['training'] and audit==results['training_audit']
    digest=sha(run/'final.pt');assert digest==training['checkpoint_sha256']==audit['checkpoint_sha256']
    assert training['iterations']==protocol['iterations']==config['iterations']
    assert training['environment_steps']==protocol['training_environment_steps']
    for name,value in config['implementation_sha256'].items():assert sha(run/Path(name).name)==value
    assert training['shared_tensors_unchanged'] and training['deployable_roundtrip'] and not training['smoke_test']
    for key in ['only_adapter_network_weights_changed','metrics_and_stateful_call_counts_consistent',
                'source_and_sequence_physics_match','failure_terms_preserved','command_and_reset_match_sequence']:
        assert audit[key]
    assert protocol['checkpoint_selection']=='final_iteration' and not protocol['automatic_promotion']
    report=dict(stage='completed',protocol=protocol,training=training,training_audit=audit,native={},sequences={},
                runtime_sources_match_now=not changed,later_historical_source_changes=changed,
                original_training_source_snapshots_verified=True,raw_summary_sha256={})
    if args.tag=='planner_post_pass_latent_ppo_low_noise_r1':
        assert config['exploration_std']==protocol['exploration_std']==.05
        assert config['anchor_weight']==protocol['anchor_weight']==.000625
        assert config['target_kl']==protocol['target_kl']==.03
        assert abs(config['anchor_weight']/(2*config['exploration_std']**2)-protocol['absolute_anchor_coefficient'])<1e-12
    if args.tag.endswith('budget_r1'):
        assert results['first_100_iterations_exact_except_elapsed']
        report['first_100_iterations_exact_except_elapsed']=True
    for skill in ['pass','shoot']:
        dest=runs/f'{args.tag}_{skill}_eval';reference=runs/f'planner_post_pass_latent_adapter_zero_r1_{skill}_eval'
        native=read(dest/'summary.json');assert native==results['native'][skill]['summary']
        assert native['requested']==native['completed']==16 and not native['expert_assisted_oracle']
        assert native['post_pass_latent_adapter_active_control_steps']==0
        assert native['initial_state_sha256']==read(reference/'summary.json')['initial_state_sha256']
        for name in ['episodes.json','first_kick_velocity.json']:assert read(dest/name)==read(reference/name)
        report['native'][skill]=dict(summary=native,episodes_and_first_impacts_exact=True)
        report['raw_summary_sha256'][dest.name]=sha(dest/'summary.json')
    assert protocol['evaluation_seeds']==[816201,816202]
    assert set(results['sequences'])=={str(seed) for seed in protocol['evaluation_seeds']}
    totals={label:{} for label in ['control','candidate']}
    for seed in protocol['evaluation_seeds']:
        pair=results['sequences'][str(seed)];assert all(pair['audit'].values())
        assert set(pair['models'])==set(totals)
        entry=dict(audit=pair['audit'],models={})
        for label,record in pair['models'].items():
            dest=runs/f'{args.tag}_{seed}_{label}';summary=read(dest/'summary.json')
            assert summary==record['summary'] and summary['requested']==32 and summary['steps']==1000
            assert summary['one_policy_instance'] and not summary['expert_loaded'] and summary['failed_lane_control']=='zero'
            assert summary['checkpoint_sha256']==(digest if label=='candidate' else protocol['control_sha256'])
            scored=outcomes(summary);assert scored==record['outcomes']
            entry['models'][label]=dict(outcomes=scored,execution_audits=record['execution_audits'])
            report['raw_summary_sha256'][dest.name]=sha(dest/'summary.json')
            for key,value in scored.items():
                if isinstance(value,int):totals[label][key]=totals[label].get(key,0)+value
                elif isinstance(value,dict):
                    target=totals[label].setdefault(key,{})
                    for k,v in value.items():target[k]=target.get(k,0)+v
        report['sequences'][str(seed)]=entry
    assert totals==results['totals']
    control=totals['control'];candidate=totals['candidate']
    passed=(candidate['full_success']>control['full_success'] and candidate['precise_shot']>=control['precise_shot'] and candidate['survivors']>=control['survivors'])
    assert passed==results['development_screen_passed']
    report.update(totals=totals,development_screen_passed=passed,automatic_promotion=False,
        scope='All original evaluation environments remain in denominators. Training counters are reported separately from closed-loop results. Reused development seeds do not establish generalization.')
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(totals=totals,development_screen_passed=passed,later_historical_source_changes=changed),indent=2))


if __name__=='__main__':main()
