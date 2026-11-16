"""Validate training-only inverse labels and weight uncertain solver results."""
import hashlib
import json
from pathlib import Path
import torch


def tensor_sha(value):
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def confidence(residual, scale=.001):
    if scale <= 0 or not torch.isfinite(residual).all() or (residual < 0).any():
        raise ValueError('Invalid solver residual or confidence scale')
    return 1 / (1 + residual / scale)


def validate_labels(record, x, contact, skill, metadata):
    """Reject mismatched source, row order, split, or invalid numeric labels."""
    assert record['metadata'] == metadata
    assert metadata['purpose'] == 'train_inverse_from_predicted_targets_v1'
    assert metadata['split'] == 'first_three_quarters_of_each_source_environment_set'
    assert metadata['expert_targets_used'] is False and metadata['validation_states_used'] is False
    assert record['skill'] == skill and record['x_sha256'] == tensor_sha(x)
    assert contact.dtype == record['contact'].dtype == torch.bool
    assert torch.equal(record['contact'], contact)
    for name, shape in [('z',(len(x),256)),('desired_targets',(len(x),29)),
                        ('initial_mapping_mse',(len(x),)),('final_mapping_mse',(len(x),))]:
        value=record[name]
        assert value.shape == shape and torch.isfinite(value).all(), name
    assert (record['initial_mapping_mse'] >= 0).all() and (record['final_mapping_mse'] >= 0).all()
    torch.testing.assert_close(record['z'].norm(dim=-1), torch.full((len(x),),16.),atol=3e-6,rtol=1e-5)


def load_labels(directory, training, provenance, source_sha, base_sha, predictor_sha):
    directory=Path(directory)
    complete=json.loads((directory/'complete.json').read_text())
    metadata=complete['metadata']
    assert metadata['source_checkpoint_sha256']==source_sha
    assert metadata['base_sha256']==base_sha and metadata['predictor_sha256']==predictor_sha
    assert metadata['data']==provenance
    labels={};audit={}
    for skill,(x,_,contact) in training.items():
        path=directory/f'{skill}.pt'
        checksum=hashlib.sha256(path.read_bytes()).hexdigest()
        assert checksum==complete['skills'][skill]['label_sha256']
        record=torch.load(path,map_location='cpu',weights_only=True)
        validate_labels(record,x,contact,skill,metadata)
        assert complete['skills'][skill]['states']==len(x)
        assert complete['skills'][skill]['state_sha256']==record['x_sha256']
        groups={}
        for name,mask in [('all',torch.ones(len(x),dtype=torch.bool)),('contact',contact),('other',~contact)]:
            residual=record['final_mapping_mse'][mask]
            if len(residual):
                groups[name]=dict(states=len(residual),mean=float(residual.mean()),
                    quantiles=dict(zip(['p50','p90','p99','max'],torch.quantile(residual,torch.tensor([.5,.9,.99,1.])).tolist())),
                    fraction_above_001=float((residual>.001).float().mean()),mean_confidence=float(confidence(residual).mean()))
        audit[skill]=dict(label_sha256=checksum,state_sha256=record['x_sha256'],groups=groups)
        labels[skill]=record
    return labels,audit


def distillation_loss(latent_error, physical_error, reliability, batch_per_skill):
    """Normalize confidence within each skill, retaining equal skill influence."""
    assert latent_error.shape==physical_error.shape==reliability.shape
    assert latent_error.numel()%batch_per_skill==0
    values=latent_error+10*physical_error
    w=reliability.reshape(-1,batch_per_skill)
    return ((values.reshape_as(w)*w).sum(-1)/w.sum(-1).clamp_min(1e-12)).mean()
