#!/usr/bin/env python3
"""
Workload-aware GPU launcher for run_fms.sh.

Distributes jobs across available GPUs by monitoring utilization via nvidia-smi.
Launches all combinations of (model configs) x (sequence lengths) across GPUs.

Input files:
  models.txt — one model config per line:
      fms_name  disp_name  model_dir  tokenizer  benchmark
  seq_lengths.txt — one sequence length per line:
      4096
      8192
      ...

Usage:
    python vela_node_launcher.py models.txt seq_lengths.txt --batch-size 4

Example models.txt:
    mamba_9.8b  bamba-32k       /gpfs/models/bamba-32k   llama3  synthetic
    mamba_9.8b  bamba-32k-ua    /gpfs/models/bamba-ua    llama3  synthetic
"""

import argparse
import subprocess
import time
import os
import sys
from datetime import datetime, timedelta
from itertools import product


def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_models_file(path):
    """Parse models file. Each line: fms_name disp_name model_dir tokenizer benchmark"""
    configs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 5:
                print(f"WARNING: skipping malformed line (expected 5 fields, got {len(parts)}): {line}")
                continue
            configs.append(tuple(parts))
    return configs


def parse_seq_lengths_file(path):
    """Parse sequence lengths file. One integer per line."""
    lengths = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            lengths.append(line)
    return lengths


def get_free_gpus(mem_threshold_mb=1000, util_threshold=10):
    """Return list of GPU IDs that are below memory and utilization thresholds."""
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []

    free = []
    for line in result.stdout.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        idx, mem_used, gpu_util = int(parts[0]), float(parts[1]), float(parts[2])
        if mem_used < mem_threshold_mb and gpu_util < util_threshold:
            free.append(idx)
    return free


def check_finished(running, job_meta):
    """Check for finished jobs, print completion info, return list of freed GPU IDs."""
    freed = []
    for gpu_id, (proc, _fout) in list(running.items()):
        if proc.poll() is not None:
            rc = proc.returncode
            meta = job_meta[id(proc)]
            elapsed = time.time() - meta["start_time"]
            duration = str(timedelta(seconds=int(elapsed)))
            status = "OK" if rc == 0 else f"FAILED (rc={rc})"
            print(f"[{timestamp()}] [GPU {gpu_id}] DONE  {meta['label']}  "
                  f"status={status}  duration={duration}", flush=True)
            _fout.close()
            del running[gpu_id]
            freed.append(gpu_id)
    return freed


def wait_for_gpu(running, job_meta, mem_threshold_mb, util_threshold, poll_interval):
    """Wait until a GPU becomes available."""
    while True:
        check_finished(running, job_meta)

        free = get_free_gpus(mem_threshold_mb, util_threshold)
        for gpu_id in free:
            if gpu_id not in running:
                return gpu_id

        time.sleep(poll_interval)


def main():
    parser = argparse.ArgumentParser(description="Distribute run_fms.sh jobs across GPUs")
    parser.add_argument("models_file", type=str,
                        help="Text file with model configs (one per line: fms_name disp_name model_dir tokenizer benchmark)")
    parser.add_argument("seq_lengths_file", type=str,
                        help="Text file with sequence lengths (one per line)")
    parser.add_argument("--batch-size", type=str, default="1")
    parser.add_argument("--mem-threshold", type=float, default=1000,
                        help="GPU memory usage (MB) below which a GPU is considered free")
    parser.add_argument("--util-threshold", type=float, default=10,
                        help="GPU utilization (%%) below which a GPU is considered free")
    parser.add_argument("--poll-interval", type=float, default=10,
                        help="Seconds between polling for free GPUs")
    parser.add_argument("--script", type=str, default=None,
                        help="Path to run_fms.sh (default: same directory as this script)")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Directory for log files (default: same directory as this script)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    run_script = args.script or os.path.join(script_dir, "run_fms.sh")
    log_dir = args.log_dir or script_dir

    if not os.path.exists(run_script):
        print(f"Error: {run_script} not found")
        sys.exit(1)

    models = parse_models_file(args.models_file)
    seq_lengths = parse_seq_lengths_file(args.seq_lengths_file)

    if not models:
        print(f"Error: no model configs found in {args.models_file}")
        sys.exit(1)
    if not seq_lengths:
        print(f"Error: no sequence lengths found in {args.seq_lengths_file}")
        sys.exit(1)

    # Build all (seq_len, model) combinations — iterate models first, then seq lengths
    all_jobs = list(product(seq_lengths, models))
    print(f"[{timestamp()}] {len(models)} model(s) x {len(seq_lengths)} seq length(s) = {len(all_jobs)} total jobs", flush=True)
    for i, (seq_len, (fms_name, disp_name, model_dir, tokenizer, benchmark)) in enumerate(all_jobs):
        print(f"  [{i+1}] {disp_name} seq={seq_len}", flush=True)
    print(flush=True)

    os.makedirs(log_dir, exist_ok=True)

    running = {}    # gpu_id -> (proc, file_handle)
    job_meta = {}   # id(proc) -> {"label": str, "start_time": float}
    launcher_start = time.time()

    for seq_len, (fms_name, disp_name, model_dir, tokenizer, benchmark) in all_jobs:
        gpu_id = wait_for_gpu(running, job_meta,
                              args.mem_threshold, args.util_threshold, args.poll_interval)

        label = f"{disp_name} seq={seq_len}"
        cmd = ["bash", run_script,
               fms_name, disp_name, model_dir,
               tokenizer, benchmark, seq_len, args.batch_size]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        log_file = os.path.join(log_dir, f"{disp_name}_seq{seq_len}_gpu{gpu_id}.log")
        fout = open(log_file, "w")

        print(f"[{timestamp()}] [GPU {gpu_id}] START {label}  (log: {log_file})", flush=True)
        proc = subprocess.Popen(cmd, env=env, stdout=fout, stderr=subprocess.STDOUT, cwd=script_dir)
        running[gpu_id] = (proc, fout)
        job_meta[id(proc)] = {"label": label, "start_time": time.time()}

        time.sleep(2)

    # Wait for all remaining jobs
    print(f"\n[{timestamp()}] All jobs launched. Waiting for completion...", flush=True)
    while running:
        check_finished(running, job_meta)
        if running:
            time.sleep(args.poll_interval)

    total_elapsed = str(timedelta(seconds=int(time.time() - launcher_start)))
    print(f"\n[{timestamp()}] All jobs complete. Total wall time: {total_elapsed}", flush=True)


if __name__ == "__main__":
    main()
