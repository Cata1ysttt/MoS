"""Original shooting ONNX diagnostic, explicitly separate from deployable policy.

Advance once per control step with the same v4 observation as the planner,
including passing steps. Output is raw environment-order expert action for
shooting rows only; callers must preserve the existing action contract and
call reset(done). No expert is loaded through load_soccer_policy.
"""
from pathlib import Path
import numpy as np
import torch
from .new_ball_task_history import NewBallTaskHistory


class ShootingExpertDiagnostic:
    def __init__(self,checkpoint=None,history_mode='raw',*,session=None):
        if history_mode not in ('raw','initialize_shoot_ball_task'):
            raise ValueError('Unknown shooting expert history mode')
        if (checkpoint is None)==(session is None):
            raise ValueError('Supply exactly one checkpoint or explicit diagnostic session')
        if session is None:
            import onnxruntime as ort
            options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
            session=ort.InferenceSession(str(Path(checkpoint)),sess_options=options,providers=['CPUExecutionProvider'])
        self.session=session;self.history_mode=history_mode
        self.history=NewBallTaskHistory().eval() if history_mode!='raw' else None
        self.records=[]

    def reset(self,env_ids=None):
        if self.history is not None:self.history.reset(env_ids)

    @torch.no_grad()
    def act(self,x):
        if x.ndim!=2 or x.shape[-1]!=2406 or x.dtype!=torch.float32:
            raise ValueError('Shooting expert diagnostic requires float32 v4 observations')
        conditioned=self.history(x) if self.history is not None else x
        selected=(x[:,-2:].argmax(-1)==1).nonzero().flatten()
        # Preserve raw ONNX values without a planner clamp, reorder or affine map.
        action=torch.zeros((len(x),29),device=x.device,dtype=x.dtype)
        for i in selected.tolist():
            inputs=dict(obs=conditioned[i:i+1,1200:2230].detach().cpu().numpy(),
                rma_obs=conditioned[i:i+1,2230:2380].detach().cpu().numpy(),time_step=np.zeros((1,1),np.float32))
            value=self.session.run(['actions'],inputs)[0]
            if value.shape!=(1,29) or not np.isfinite(value).all():
                raise ValueError('Invalid original shooting expert action')
            action[i]=torch.as_tensor(value[0],device=x.device,dtype=x.dtype)
        rma=task=0
        if self.history is not None:
            age=self.history.age;rma=int(((age>0)&(age<=50)).sum());task=int(((age>0)&(age<=10)).sum())
        self.records.append(dict(shoot_environments=len(selected),history_mode=self.history_mode,
            new_ball_history_environments=rma,new_ball_task_history_environments=task))
        return action
