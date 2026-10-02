"""Run and summarize all main experiments without editing source JSON files."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = (
    "voc2007_baseline.json", "voc2007_feature.json",
    "coco17_baseline.json", "coco17_feature.json",
    "nuswide_baseline.json", "nuswide_feature.json",
)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def mean_std(values):
    values = list(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return statistics.mean(values), std


def parse_metrics(log_path):
    metrics = {}
    pattern = re.compile(r"^([^:]+):\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$")
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = pattern.fullmatch(line.strip())
        if match:
            metrics[match.group(1)] = float(match.group(2))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    parser.add_argument("--backbones", nargs="+", choices=["resnet50", "resnet101"], default=["resnet50", "resnet101"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--episodes", type=int, help="Override training episodes; omit for JSON value (formal run).")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    output = args.output_dir or ROOT / "experiment_results" / ("main_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_paths = [ROOT / "configs" / name for name in CONFIGS]
    before_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    jobs = [(path, backbone, seed) for path in source_paths for backbone in args.backbones for seed in args.seeds]
    manifest = {
        "kind": "main", "created_at": datetime.now().isoformat(), "device": args.device,
        "workers": args.workers, "seeds": args.seeds, "backbones": args.backbones,
        "training_episode_override": args.episodes, "source_config_sha256_before": before_hashes,
        "runs": [],
    }
    write_json(output / "manifest.json", manifest)
    environment = dict(os.environ, PYTHONUNBUFFERED="1")
    if args.device == "cpu":
        environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")

    def run(source_path, backbone, seed):
        source = json.loads(source_path.read_text(encoding="utf-8"))
        effective = json.loads(json.dumps(source))
        effective["seed"] = seed
        effective["dataset"]["backbone"] = backbone
        if args.episodes is not None:
            effective["training"]["episodes"] = args.episodes
        dataset = source_path.stem.split("_")[0]
        model = "baseline" if source_path.stem.endswith("baseline") else "ourmodel"
        tag = f"{dataset}_{model}_{backbone}_seed{seed}"
        run_dir = output / tag
        run_dir.mkdir()
        checkpoint = run_dir / "best.pt"
        effective["training"]["checkpoint_path"] = str(checkpoint)
        effective_path = run_dir / "effective_config.json"
        write_json(effective_path, effective)
        train_command = [sys.executable, "-u", str(ROOT / "scripts" / "train.py"), "--config", str(effective_path), "--device", args.device]
        test_command = [sys.executable, "-u", str(ROOT / "scripts" / "evaluate.py"), "--config", str(effective_path), "--checkpoint", str(checkpoint), "--phase", "test", "--device", args.device]
        row = {
            "tag": tag, "dataset": dataset, "model": model, "backbone": backbone, "seed": seed,
            "training_episodes": effective["training"]["episodes"], "status": "running",
            "run_dir": str(run_dir.relative_to(ROOT)), "commands": [train_command, test_command],
        }
        started = time.monotonic()
        for stage, command in (("train", train_command), ("test", test_command)):
            with (run_dir / f"{stage}.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                row.update(status="failed", failed_stage=stage, returncode=result.returncode)
                write_json(run_dir / "result.json", row)
                return row
        metrics = parse_metrics(run_dir / "test.log")
        row.update(status="ok" if "mAP" in metrics else "failed", metrics=metrics, elapsed_seconds=time.monotonic() - started)
        write_json(run_dir / "result.json", row)
        return row

    print(f"Output: {output}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = [executor.submit(run, *job) for job in jobs]
        for future in as_completed(pending):
            row = future.result()
            manifest["runs"].append(row)
            write_json(output / "manifest.json", manifest)
            print(f"[{len(manifest['runs'])}/{len(jobs)}] {row['tag']}: {row['status']} mAP={row.get('metrics', {}).get('mAP')}", flush=True)

    after_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    manifest["source_config_sha256_after"] = after_hashes
    manifest["source_configs_unchanged"] = before_hashes == after_hashes
    write_json(output / "manifest.json", manifest)
    if not manifest["source_configs_unchanged"]:
        raise RuntimeError("A source config changed during the experiment")
    failed = [row for row in manifest["runs"] if row["status"] != "ok"]
    if failed:
        raise RuntimeError(f"{len(failed)} runs failed; inspect manifest.json and per-run logs")

    rows = manifest["runs"]
    report = [
        "# 主要实验结果", "",
        f"种子：{', '.join(map(str, args.seeds))}；backbone：{', '.join(args.backbones)}；设备：{args.device}。",
        "mAP 为逐 episode 计算后平均；表中 ± 为跨随机种子的样本标准差。模型按固定验证 episodes 的 mAP 选择，测试阈值来自验证集。", "",
        "| 数据集 | Backbone | Baseline mAP | OurModel mAP | 配对提升 | OurModel胜出种子 |", "|---|---|---:|---:|---:|---:|",
    ]
    for dataset in ("voc2007", "coco17", "nuswide"):
        for backbone in args.backbones:
            by_model = {
                model: {row["seed"]: row["metrics"]["mAP"] for row in rows if row["dataset"] == dataset and row["backbone"] == backbone and row["model"] == model}
                for model in ("baseline", "ourmodel")
            }
            common = sorted(set(by_model["baseline"]) & set(by_model["ourmodel"]))
            baseline_values = [by_model["baseline"][seed] for seed in common]
            our_values = [by_model["ourmodel"][seed] for seed in common]
            differences = [our - baseline for baseline, our in zip(baseline_values, our_values)]
            bm, bs = mean_std(baseline_values); om, osd = mean_std(our_values); dm, ds = mean_std(differences)
            report.append(f"| {dataset} | {backbone} | {bm:.6f} ± {bs:.6f} | {om:.6f} ± {osd:.6f} | {dm:+.6f} ± {ds:.6f} | {sum(value > 0 for value in differences)}/{len(differences)} |")
    report.extend(["", "## 单次运行", "", "| 数据集 | 模型 | Backbone | Seed | mAP | Micro-F1 | Macro-F1 |", "|---|---|---|---:|---:|---:|---:|"])
    for row in sorted(rows, key=lambda item: (item["dataset"], item["backbone"], item["model"], item["seed"])):
        metrics = row["metrics"]
        report.append(f"| {row['dataset']} | {row['model']} | {row['backbone']} | {row['seed']} | {metrics['mAP']:.6f} | {metrics['Micro-F1']:.6f} | {metrics['Macro-F1']:.6f} |")
    (output / "main_experiment_results.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Report: {output / 'main_experiment_results.md'}", flush=True)


if __name__ == "__main__":
    main()
