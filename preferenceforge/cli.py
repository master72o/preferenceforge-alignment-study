"""
CLI Interface for PreferenceForge: Alignment & Calibration Studio.
"""

import os
import sys
import argparse
from preferenceforge.core.schema import PreferenceBenchmarkDataset
from preferenceforge.core.evaluator import PreferenceForgeEvaluator
from preferenceforge.core.bradley_terry import BradleyTerryEngine
from preferenceforge.core.length_bias import LengthBiasDetector
from preferenceforge.training.dpo_trainer import build_parser as build_train_parser, run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="preferenceforge",
        description="PreferenceForge: Bradley-Terry Modeling, DPO Training & Length-Bias Alignment Studio",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available commands")

    # Benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run complete alignment evaluation suite across all checkpoints")
    bench_parser.add_argument("--dataset", default="data/preference_dataset.jsonl", help="Path to JSONL preference dataset")
    bench_parser.add_argument("--output-dir", default="results", help="Directory for JSON and CSV exports")
    bench_parser.add_argument("--report-path", default="reports/preference_reward_report.md", help="Path for Markdown report")

    # Calibrate
    calib_parser = subparsers.add_parser("calibrate", help="Evaluate ECE and compute optimal temperature scaling")
    calib_parser.add_argument("--dataset", default="data/preference_dataset.jsonl", help="Path to JSONL preference dataset")
    calib_parser.add_argument("--model", default="RM_v3_Margin_Scaled", help="Model checkpoint key")

    # Bias Audit
    bias_parser = subparsers.add_parser("bias-audit", help="Audit verbosity and length bias across models")
    bias_parser.add_argument("--dataset", default="data/preference_dataset.jsonl", help="Path to JSONL preference dataset")
    bias_parser.add_argument("--model", default="RM_v1_Baseline", help="Model checkpoint key")

    # Train DPO
    train_parser = subparsers.add_parser("train", help="Run or dry-run DPO LoRA training pipeline")
    train_parser.add_argument("--model-name", default="HuggingFaceTB/SmolLM-135M-Instruct", help="Hugging Face model ID")
    train_parser.add_argument("--dataset-path", default="data/preference_dataset.jsonl", help="Path to dataset")
    train_parser.add_argument("--output-dir", default="checkpoints/dpo_adapter", help="Adapter output directory")
    train_parser.add_argument("--beta", type=float, default=0.1, help="DPO beta regularization")
    train_parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    train_parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    train_parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Gradient accumulation steps")
    train_parser.add_argument("--max-steps", type=int, default=50, help="Max steps")
    train_parser.add_argument("--lora-r", type=int, default=16, help="LoRA r")
    train_parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    train_parser.add_argument("--use-4bit", action="store_true", help="4-bit quantization")
    train_parser.add_argument("--dry-run", action="store_true", help="Simulate pipeline setup")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.subcommand == "benchmark":
        print(f"[PreferenceForge] Loading dataset from: {args.dataset}")
        dataset = PreferenceBenchmarkDataset.from_jsonl(args.dataset)
        print(f"[PreferenceForge] Loaded {len(dataset)} preference pairs across {len(dataset.domains())} domains.")

        evaluator = PreferenceForgeEvaluator(dataset)
        summary = evaluator.run_full_suite()

        print("\n" + "=" * 90)
        print(f"{'Checkpoint':<24} | {'Accuracy':<9} | {'Mean Margin':<12} | {'ECE':<8} | {'Discordant Acc':<14} | {'Length Corr (r)':<15}")
        print("=" * 90)
        for key, name, _ in PreferenceForgeEvaluator.CHECKPOINTS:
            m = summary.models[key]
            b = summary.length_bias[key]
            print(
                f"{key:<24} | {m.accuracy*100:>7.1f}% | {m.mean_margin:>+11.3f} | {m.expected_calibration_error*100:>6.1f}% | "
                f"{b.discordant_accuracy*100:>13.1f}% | {b.pearson_correlation:>+14.3f}"
            )
        print("=" * 90)

        # Export outputs
        os.makedirs(args.output_dir, exist_ok=True)
        summary_json_path = os.path.join(args.output_dir, "preferenceforge_summary.json")
        csv_path = os.path.join(args.output_dir, "preferenceforge_predictions.csv")
        evaluator.export_summary_json(summary, summary_json_path)
        evaluator.export_predictions_csv(csv_path)
        report = evaluator.generate_markdown_report(summary, args.report_path)

        print(f"\n[PreferenceForge] Exported summary to: {summary_json_path}")
        print(f"[PreferenceForge] Exported CSV to: {csv_path}")
        print(f"[PreferenceForge] Exported report to: {args.report_path}")
        print("\nBenchmark completed successfully!")

    elif args.subcommand == "calibrate":
        print(f"[PreferenceForge Calibrate] Loading dataset: {args.dataset}")
        dataset = PreferenceBenchmarkDataset.from_jsonl(args.dataset)
        metrics = BradleyTerryEngine.evaluate_model(dataset.records, args.model, optimize_temp=True)

        print(f"\nModel: {args.model}")
        print(f"  Total Samples : {metrics.total_samples}")
        print(f"  Accuracy      : {metrics.accuracy*100:.2f}%")
        print(f"  Mean Margin   : {metrics.mean_margin:+.4f}")
        print(f"  Expected Calibration Error (ECE) : {metrics.expected_calibration_error*100:.2f}%")
        print(f"  MCE           : {metrics.max_calibration_error*100:.2f}%")
        print(f"  Brier Score   : {metrics.brier_score:.4f}")
        print(f"  Optimal Temp  : T* = {metrics.optimal_temperature:.4f}")

        print("\nCalibration Bins (Equal Width):")
        for b in metrics.bins:
            if b.count > 0:
                print(f"  Bin {b.bin_index} [{b.lower_bound:.2f}-{b.upper_bound:.2f}]: count={b.count:2d}, conf={b.avg_confidence:.3f}, acc={b.empirical_accuracy:.3f}, gap={b.calibration_gap:.3f}")

    elif args.subcommand == "bias-audit":
        print(f"[PreferenceForge Bias-Audit] Loading dataset: {args.dataset}")
        dataset = PreferenceBenchmarkDataset.from_jsonl(args.dataset)
        bias = LengthBiasDetector.analyze_length_bias(dataset.records, args.model)

        print(f"\nModel: {args.model}")
        print(f"  Pearson Length Corr (r) : {bias.pearson_correlation:+.4f} (p={bias.p_value:.4f})")
        print(f"  Spearman Rank Corr (rho): {bias.spearman_correlation:+.4f}")
        print(f"  Concordant Accuracy     : {bias.concordant_accuracy*100:.1f}% (count={bias.concordant_count})")
        print(f"  Discordant Accuracy     : {bias.discordant_accuracy*100:.1f}% (count={bias.discordant_count})")
        print(f"  Neutral Accuracy        : {bias.neutral_accuracy*100:.1f}% (count={bias.neutral_count})")
        print(f"  Verbosity Bias Flag     : {'TRIGGERED' if bias.verbosity_bias_flag else 'CLEAN'}")
        print(f"  Severity Level          : {bias.bias_severity.upper()}")

    elif args.subcommand == "train":
        run_training(args)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
