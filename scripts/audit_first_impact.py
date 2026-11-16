"""Summarize observed ball velocities without replacing the environment's task metrics."""
import argparse
import json
import statistics
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--evaluations', type=Path, nargs='+', required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
result = {}
for path in args.evaluations:
    summary = json.loads((path / 'summary.json').read_text())
    source = json.loads((path / 'first_kick_velocity.json').read_text())
    rows = list(source['samples'].values())
    result[path.name] = dict(
        source=str(path), sampling=source['sampling'],
        expert_assisted=summary['expert_assisted_oracle'],
        completed=summary['completed'], requested=summary['requested'],
        observed_moving_ball_samples=len(rows),
        samples_with_direction_error_at_most_10deg=sum(row['direction_error_deg'] <= 10 for row in rows),
        median_speed_mps=statistics.median(row['speed_mps'] for row in rows) if rows else None,
        median_forward_speed_mps=statistics.median(row['forward_speed_mps'] for row in rows) if rows else None,
        median_target_speed_mps=statistics.median(row['target_speed_mps'] for row in rows) if rows else None,
        task_quality_kick_success=summary['quality_kick_success'],
        task_quality_direction_success=summary['quality_ball_direction_success_10deg'],
    )
args.output.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
