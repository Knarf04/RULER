import argparse
import ast
import os
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


def parse_log(log_path):
    """Parse a single exp.log file.

    Returns dict: {threshold: (total_pruned, total_tokens)} aggregated
    across all layers and all samples.
    """
    # threshold -> (total_pruned, total_tokens)
    agg = defaultdict(lambda: [0, 0])

    with open(log_path) as f:
        for line in f:
            # Format: "layer_name @0.001: [n1, n2, ...]/seq_len"
            m = re.match(r'.+ @([\d.]+): (.+)/(\d+)$', line.strip())
            if not m:
                continue
            thresh = float(m.group(1))
            pruned_per_head = ast.literal_eval(m.group(2))
            seq_len = int(m.group(3))

            agg[thresh][0] += sum(pruned_per_head)
            agg[thresh][1] += len(pruned_per_head) * seq_len

    results = {}
    for thresh, (pruned, total) in sorted(agg.items()):
        results[thresh] = pruned / total if total > 0 else None
    return results


def main():
    parser = argparse.ArgumentParser(description="Parse prune stats from exp.log files")
    parser.add_argument("--data_dir", type=Path, required=True,
                        help="Directory containing *_exp.log files (same as PRED_DIR)")
    args = parser.parse_args()

    log_files = sorted(args.data_dir.glob("*_exp.log"))
    if not log_files:
        print("No *_exp.log files found.")
        return

    # Collect results: {task: {threshold: ratio}}
    all_results = {}
    all_thresholds = set()
    for log_file in log_files:
        task = log_file.stem.removesuffix("_exp")
        results = parse_log(log_file)
        all_results[task] = results
        all_thresholds.update(results.keys())

    tasks = list(all_results.keys())
    thresholds = sorted(all_thresholds)

    rows = [['Tasks'] + tasks]
    for thresh in thresholds:
        row = [f'Sparsity@{thresh}']
        for task in tasks:
            ratio = all_results[task].get(thresh)
            row.append(f"{ratio:.4%}" if ratio is not None else "N/A")
        rows.append(row)

    output_file = os.path.join(args.data_dir, 'summary_sparsity.csv')
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False, sep='\t')
    print(df)
    print(f'\nSaved sparsity results to {output_file}')


if __name__ == "__main__":
    main()
