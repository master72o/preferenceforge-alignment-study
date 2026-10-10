"""
CLI Interface for Preference Modeling & Reward Model Calibration.
"""

import argparse
import sys
from preference_reward.schema import PreferenceDataset, RewardModelVersion
from preference_reward.experiment_runner import ExperimentRunner
from preference_reward.metrics import MetricsCalculator
from preference_reward.visualizer import Visualizer
from preference_reward.report_generator import ReportGenerator


def main():
    parser = argparse.ArgumentParser(description="Preference Modeling & Reward Calibration CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    run_parser = subparsers.add_parser("run", help="Run reward model calibration benchmark")
    run_parser.add_argument("--dataset", required=True, help="Path to JSONL preference dataset")
    run_parser.add_argument("--output-dir", default="results", help="Directory to save JSON/CSV outputs")
    run_parser.add_argument("--report-path", default="reports/preference_reward_report.md", help="Path to save Markdown report")
    run_parser.add_argument("--figures-dir", default="reports/figures", help="Directory to save figure plots")

    args = parser.parse_args()

    if args.command == "run":
        print(f"Loading preference dataset from: {args.dataset}")
        dataset = PreferenceDataset.from_jsonl(args.dataset)
        print(f"Loaded {len(dataset.pairs)} preference pairs.")

        print("Evaluating reward predictions across RM checkpoints (v1 Baseline, v2 DPO Calibrated, v3 Margin Scaled)...")
        all_predictions = {rm_v: [] for rm_v in RewardModelVersion}

        for pair in dataset.pairs:
            res_dict = ExperimentRunner.evaluate_pair_across_checkpoints(pair)
            for rm_v, pred in res_dict.items():
                all_predictions[rm_v].append(pred)

        metrics = MetricsCalculator.calculate_suite_metrics(all_predictions)
        print("\nReward Model Calibration Summary:")
        for rm_v, m in metrics.items():
            print(f"  {rm_v:22s}: Accuracy={m['accuracy']*100:.1f}%, ECE={m['expected_calibration_error']*100:.1f}%, Mean Margin (Delta r)={m['mean_margin']:+.4f}")

        print(f"\nExporting results to: {args.output_dir}")
        ReportGenerator.export_results(all_predictions, args.output_dir)

        print(f"Generating visualizations in: {args.figures_dir}")
        Visualizer.generate_all_figures(metrics, all_predictions, args.figures_dir)

        print(f"Generating research report at: {args.report_path}")
        ReportGenerator.generate_markdown_report(metrics, args.report_path)

        print("\nReward model evaluation completed successfully!")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
