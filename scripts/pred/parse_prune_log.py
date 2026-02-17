import argparse
import ast
import os
import pandas as pd
from pathlib import Path


def parse_log(log_path, phase="decode"):
    total_pruned = 0
    total_tokens = 0
    tag = f"[{phase}]:"

    with open(log_path) as f:
        for line in f:
            if tag not in line:
                continue
            # Format: "layer_name [decode]: [n1, n2, ...]/cache_len"
            stats_part = line.split(tag)[1].strip()
            list_str, cache_len_str = stats_part.rsplit("/", 1)
            pruned_per_head = ast.literal_eval(list_str)
            cache_len = int(cache_len_str)

            total_pruned += sum(pruned_per_head)
            total_tokens += len(pruned_per_head) * cache_len

    if total_tokens == 0:
        return None
    return total_pruned / total_tokens


def main():
    parser = argparse.ArgumentParser(description="Parse prune stats from exp.log files")
    parser.add_argument("--data_dir", type=Path, required=True,
                        help="Directory containing *_exp.log files (same as PRED_DIR)")
    args = parser.parse_args()

    log_files = sorted(args.data_dir.glob("*_exp.log"))

    tasks = []
    decode_sparsity = []
    prefill_sparsity = []

    for log_file in log_files:
        task = log_file.stem.removesuffix("_exp")
        tasks.append(task)

        ratio = parse_log(log_file, phase="decode")
        decode_sparsity.append(f"{ratio:.4%}" if ratio is not None else "N/A")

        ratio = parse_log(log_file, phase="prefill")
        prefill_sparsity.append(f"{ratio:.4%}" if ratio is not None else "N/A")

    if not tasks:
        print("No *_exp.log files found.")
        return

    dfs = [
        ['Tasks'] + tasks,
        ['Prefill Sparsity'] + prefill_sparsity,
        ['Decode Sparsity'] + decode_sparsity,
    ]

    output_file = os.path.join(args.data_dir, 'summary_sparsity.csv')
    df = pd.DataFrame(dfs)
    df.to_csv(output_file, index=False, sep='\t')
    print(df)
    print(f'\nSaved sparsity results to {output_file}')


if __name__ == "__main__":
    main()
