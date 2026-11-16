"""Refresh a reviewable report from completed evaluations and live job status."""
import argparse,json,os,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[1];runs=root/'runs';report=root/'reports/joint_v2_debug_20261113.md'
marker='\n## 自动更新的实测结果\n'
def refresh():
    status=json.loads((runs/'joint_v2_status.json').read_text())
    lines=[marker,'\n记录更新时间（Unix 秒）：'+str(int(time.time()))+'。\n',f"调度状态：`{status.get('status')}`；阶段：`{status.get('stage','—')}`。\n",'\n| 策略/阶段 | 技能 | 完成 episode | 触球成功 | 方向成功 ≤10° |\n|---|---|---:|---:|---:|\n']
    names=[('专家','pass', 'pass_shoot_pass_expert_eval_v1'),('专家','shoot','pass_shoot_shoot_expert_eval_v1'),('联合 v1','pass','pass_shoot_pass_joint_eval_v1'),('联合 v1','shoot','pass_shoot_shoot_joint_eval_v1'),('原 s1 对照','pass','joint_v2_pass_s1_control'),('v2 第一射门训练阶段（未完成交替轮）','shoot','joint_v2_shoot_stage_probe')]
    names += [(f'联合 v2 第{r}轮',s,f'joint_v2_r{r}_{s}_eval') for r in range(1,4) for s in ['pass','shoot']]
    for label,skill,name in names:
        path=runs/name/'summary.json'
        if not path.exists():continue
        data=json.loads(path.read_text());n=data['completed']
        reference=json.loads((runs/f'pass_shoot_{skill}_expert_eval_v1/summary.json').read_text())
        assert data['initial_state_sha256']==reference['initial_state_sha256'], name
        assert not data.get('expert_assisted_oracle'), name
        lines.append(f"| {label} | {skill} | {n}/{data['requested']} | {data['quality_kick_success']:.1%} | {data['quality_ball_direction_success_10deg']:.1%} |\n")
    lines.append('\n只列已经生成 summary 的自主评测。训练误差、专家混合执行、latent oracle 均不作为成功率。不同任务的环境各自与对应专家配对，不表示同一环境中自动切换技能。\n')
    for skill in ['shoot','pass']:
        paths=sorted(runs.glob(f'joint_v2_r*_{skill}_train/joint_metrics.jsonl'))
        if paths:
            rows=paths[-1].read_text().splitlines()
            if rows:
                last=json.loads(rows[-1]);lines.append(f"\n最近 {skill} 训练：`{paths[-1].parent.name}`，{last['transitions']} transitions、{last['updates']} updates。\n")
    report.write_text(report.read_text().split(marker)[0]+''.join(lines))
    return status
while True:
    status=refresh()
    if not a.watch or status['status'] in ['completed','failed']:break
    try:os.kill(status['driver_pid'],0)
    except ProcessLookupError:break
    time.sleep(55)
print(json.dumps(status,ensure_ascii=False))
