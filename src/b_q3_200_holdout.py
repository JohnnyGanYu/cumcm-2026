"""Execute the preselected, unused 100+60 Q3 holdout cases exactly once."""
import json
from pathlib import Path
import subprocess
import sys

from src.b_q3_200_eval import source_hashes


def main():
    proposal = json.loads(Path('results/q3_200_holdout_seed_proposal.json').read_text(encoding='utf-8'))
    plan = {'groups': proposal['groups'], 'source_sha256': source_hashes(),
            'variants': ['baseline', 'candidate'], 'workers': 3,
            'expected_records': 320, 'selection': json.loads(
                Path('results/q3_200_selection.json').read_text(encoding='utf-8'))}
    out = Path('results/q3_200_holdout_plan.json')
    if out.exists():
        raise RuntimeError('Refusing to overwrite the final holdout plan')
    out.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    for group in plan['groups']:
        assert source_hashes() == plan['source_sha256']
        command = [sys.executable, '-m', 'src.b_q3_200_eval', '--seeds',
                   *map(str, group['seeds']), '--scenarios', f"{group['count'] or 0}:0",
                   '--variants', 'baseline', 'candidate', '--workers', '3', '--out',
                   f"results/q3_200_holdout_{group['name']}.json"]
        subprocess.run(command, check=True)
    assert source_hashes() == plan['source_sha256']


if __name__ == '__main__':
    main()
