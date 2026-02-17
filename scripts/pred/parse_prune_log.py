import argparse
import ast
import os
from pathlib import Path


def parse_log(log_path):
    total_pruned = 0
    total_tokens = 0

    with open(log_path) as f:
        for line in f:
            if "[decode]:" not in line:
                continue
            # Format: "layer_name [decode]: [n1, n2, ...]/cache_len"
            stats_part = line.split("[decode]:")[1].strip()
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
    parser.add_argument("path", type=Path, help="Path to a single *_exp.log file or a directory containing them")
    args = parser.parse_args()

    if args.path.is_file():
        log_files = [args.path]
    else:
        log_files = sorted(args.path.glob("*_exp.log"))

    for log_file in log_files:
        task = log_file.stem.removesuffix("_exp")
        ratio = parse_log(log_file)
        if ratio is not None:
            print(f"{task}: {ratio:.4%} pruned")
        else:
            print(f"{task}: no decode stats found")


if __name__ == "__main__":
    main()
