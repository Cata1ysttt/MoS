import copy
import json
import pytest
from fiber_soccer.evaluation_reference import load_reference_evaluations


def fixture_data(tmp_path, mutate=None):
    protocol=dict(seed=123,num_envs=2,checkpoint_sha256='a'*64)
    result={}
    for skill in ['pass','shoot']:
        common=dict(seed=123,requested=2,completed=2,expert_assisted_oracle=False,initial_state_sha256=skill)
        result[skill]=dict(planner=dict(common,controller='unified_latent_planner',checkpoint='planner.pt'),
                          expert=dict(common,controller='expert',checkpoint=skill+'.pt'))
    status=dict(status='completed',protocol=copy.deepcopy(protocol),results=copy.deepcopy(result))
    if mutate:mutate(protocol,result,status)
    for suffix,value in [('protocol',protocol),('results',result),('status',status)]:
        (tmp_path/('control_'+suffix+'.json')).write_text(json.dumps(value))


def test_completed_controls_are_reused_with_source_paths_and_hash(tmp_path):
    fixture_data(tmp_path)
    selected,provenance=load_reference_evaluations(tmp_path,'control',123,2,'a'*64,{'pass':'pass.pt','shoot':'shoot.pt'})
    assert selected['pass']['baseline']['checkpoint']=='planner.pt'
    assert selected['shoot']['expert']['controller']=='expert'
    assert provenance['validation_seed_previously_evaluated'] is True
    assert len(provenance['reference_results_sha256'])==64
    assert provenance['reference_runs']['pass']['baseline'].endswith('control_pass_planner')


@pytest.mark.parametrize('field,value',[
    ('completed',1),('seed',124),('expert_assisted_oracle',True),
    ('controller','expert'),('initial_state_sha256','different'),
])
def test_reject_invalid_reference_rollouts_even_when_status_matches(tmp_path,field,value):
    def mutate(protocol,result,status):
        result['pass']['planner'][field]=value
        status['results']=copy.deepcopy(result)
    fixture_data(tmp_path,mutate)
    with pytest.raises(ValueError):
        load_reference_evaluations(tmp_path,'control',123,2,'a'*64,{'pass':'pass.pt','shoot':'shoot.pt'})


@pytest.mark.parametrize('seed,count,checksum',[(124,2,'a'*64),(123,3,'a'*64),(123,2,'b'*64)])
def test_reject_different_requested_pairing(tmp_path,seed,count,checksum):
    fixture_data(tmp_path)
    with pytest.raises(ValueError):
        load_reference_evaluations(tmp_path,'control',seed,count,checksum,{'pass':'pass.pt','shoot':'shoot.pt'})


def test_reject_pending_or_rewritten_reference(tmp_path):
    fixture_data(tmp_path,lambda protocol,result,status:status.update(status='running'))
    with pytest.raises(ValueError):
        load_reference_evaluations(tmp_path,'control',123,2,'a'*64,{'pass':'pass.pt','shoot':'shoot.pt'})
    fixture_data(tmp_path,lambda protocol,result,status:result['pass']['expert'].update(completed=1))
    with pytest.raises(ValueError):
        load_reference_evaluations(tmp_path,'control',123,2,'a'*64,{'pass':'pass.pt','shoot':'shoot.pt'})


def test_reference_can_select_the_original_baseline_by_checkpoint_hash(tmp_path):
    def mutate(protocol,result,status):
        protocol['baseline_checkpoint_sha256']='b'*64
        for group in result.values():
            group['baseline']=dict(group['planner'],checkpoint='previous.pt')
        status['protocol']=copy.deepcopy(protocol)
        status['results']=copy.deepcopy(result)
    fixture_data(tmp_path,mutate)
    selected,provenance=load_reference_evaluations(tmp_path,'control',123,2,'b'*64,{'pass':'pass.pt','shoot':'shoot.pt'})
    assert selected['pass']['baseline']['checkpoint']=='previous.pt'
    assert provenance['reference_planner_role']=='baseline'
    assert provenance['reference_runs']['pass']['baseline'].endswith('control_pass_baseline')
