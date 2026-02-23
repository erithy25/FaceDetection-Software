"""
Evaluation Pipeline — Batch processing and metrics generation.

Processes the FaceForensics++ test set in batch mode, generates
per-video predictions, and outputs publication-ready metrics
including confusion matrix, ROC curve, and per-method accuracy.
"""

import os
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("silentwitness.evaluator")


class Evaluator:
    """Batch evaluation pipeline for FaceForensics++ benchmark."""

    def __init__(
        self,
        dataset_root: str,
        output_dir: str = "evaluation_results",
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.predictions: list[dict] = []

    def evaluate_dataset(
        self,
        methods: Optional[list[str]] = None,
        compression: str = "c23",
        max_videos: Optional[int] = None,
    ) -> dict:
        """Run evaluation on the FaceForensics++ dataset.

        Args:
            methods: List of manipulation methods to evaluate.
                     Default: all four (Deepfakes, Face2Face, FaceSwap, NeuralTextures).
            compression: Compression quality ("raw", "c23", "c40").
            max_videos: Maximum number of videos to process (for debugging).

        Returns:
            Dict with aggregated metrics.
        """
        if methods is None:
            methods = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]

        from detector import FaceDetector
        from classifier import DeepfakeClassifier
        from scorer import ScoreFusion

        detector = FaceDetector()
        classifier = DeepfakeClassifier()
        classifier.load_models()

        self.predictions = []
        video_count = 0

        # Process real videos
        real_dir = self.dataset_root / "original_sequences" / "youtube" / compression / "videos"
        if real_dir.exists():
            for video_path in sorted(real_dir.glob("*.mp4")):
                if max_videos and video_count >= max_videos:
                    break
                score = self._process_video(video_path, detector, classifier)
                self.predictions.append({
                    "video": video_path.name,
                    "method": "original",
                    "label": 1.0,  # Real
                    "score": score,
                })
                video_count += 1
                logger.info(f"Processed real video {video_count}: {video_path.name} -> {score:.3f}")

        # Process manipulated videos
        for method in methods:
            method_dir = (
                self.dataset_root
                / "manipulated_sequences"
                / method
                / compression
                / "videos"
            )
            if not method_dir.exists():
                logger.warning(f"Method directory not found: {method_dir}")
                continue

            for video_path in sorted(method_dir.glob("*.mp4")):
                if max_videos and video_count >= max_videos:
                    break
                score = self._process_video(video_path, detector, classifier)
                self.predictions.append({
                    "video": video_path.name,
                    "method": method,
                    "label": 0.0,  # Fake
                    "score": score,
                })
                video_count += 1
                logger.info(
                    f"Processed {method} video {video_count}: "
                    f"{video_path.name} -> {score:.3f}"
                )

        metrics = self._compute_metrics()
        self._generate_report(metrics)
        return metrics

    def _process_video(self, video_path: Path, detector, classifier) -> float:
        """Process a single video and return the average detection score."""
        cap = cv2.VideoCapture(str(video_path))
        scores = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process every 10th frame for efficiency
            if frame_idx % 10 != 0:
                frame_idx += 1
                continue

            # Resize to processing resolution
            frame = cv2.resize(frame, (640, 480))

            result = detector.detect(frame)
            if result is not None:
                face_crop, _, _ = result
                score, _ = classifier.classify_frame(face_crop)
                scores.append(score)

            frame_idx += 1

        cap.release()
        classifier.reset()
        return float(np.mean(scores)) if scores else 0.5

    def _compute_metrics(self) -> dict:
        """Compute evaluation metrics from predictions."""
        try:
            from sklearn.metrics import (
                roc_auc_score,
                accuracy_score,
                precision_score,
                recall_score,
                f1_score,
                confusion_matrix,
            )
        except ImportError:
            logger.error("scikit-learn required for metrics computation")
            return {}

        labels = np.array([p["label"] for p in self.predictions])
        scores = np.array([p["score"] for p in self.predictions])
        preds = (scores >= 0.5).astype(float)

        metrics = {
            "auc": float(roc_auc_score(labels, scores)),
            "accuracy": float(accuracy_score(labels, preds)),
            "precision": float(precision_score(labels, preds, zero_division=0)),
            "recall": float(recall_score(labels, preds, zero_division=0)),
            "f1": float(f1_score(labels, preds, zero_division=0)),
            "confusion_matrix": confusion_matrix(labels, preds).tolist(),
            "total_videos": len(self.predictions),
        }

        # Per-method accuracy
        method_metrics = {}
        for method in set(p["method"] for p in self.predictions):
            method_preds = [p for p in self.predictions if p["method"] == method]
            method_labels = np.array([p["label"] for p in method_preds])
            method_scores = np.array([p["score"] for p in method_preds])
            method_binary = (method_scores >= 0.5).astype(float)
            method_metrics[method] = {
                "accuracy": float(accuracy_score(method_labels, method_binary)),
                "count": len(method_preds),
                "avg_score": float(np.mean(method_scores)),
            }
        metrics["per_method"] = method_metrics

        return metrics

    def _generate_report(self, metrics: dict) -> None:
        """Generate evaluation report with plots and markdown summary."""
        self._plot_roc_curve(metrics)
        self._plot_confusion_matrix(metrics)
        self._write_markdown_report(metrics)

    def _plot_roc_curve(self, metrics: dict) -> None:
        """Generate ROC curve plot."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from sklearn.metrics import roc_curve

            labels = np.array([p["label"] for p in self.predictions])
            scores = np.array([p["score"] for p in self.predictions])
            fpr, tpr, _ = roc_curve(labels, scores)

            fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
            ax.plot(fpr, tpr, color="#34D399", linewidth=2, label=f"AUC = {metrics['auc']:.3f}")
            ax.plot([0, 1], [0, 1], color="#2A2A2E", linestyle="--", linewidth=1)
            ax.set_xlabel("False Positive Rate", fontsize=11)
            ax.set_ylabel("True Positive Rate", fontsize=11)
            ax.set_title("ROC Curve — SilentWitness", fontsize=13)
            ax.legend(loc="lower right", fontsize=10)
            ax.set_facecolor("#0A0A0B")
            fig.patch.set_facecolor("#0A0A0B")
            ax.tick_params(colors="#8E8E93")
            ax.xaxis.label.set_color("#F5F5F7")
            ax.yaxis.label.set_color("#F5F5F7")
            ax.title.set_color("#F5F5F7")

            plt.tight_layout()
            plt.savefig(self.output_dir / "roc_curve.png", dpi=300, facecolor="#0A0A0B")
            plt.close()
            logger.info("ROC curve saved")

        except ImportError:
            logger.warning("matplotlib required for plot generation")

    def _plot_confusion_matrix(self, metrics: dict) -> None:
        """Generate confusion matrix plot."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns

            cm = np.array(metrics.get("confusion_matrix", [[0, 0], [0, 0]]))

            fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Greens",
                xticklabels=["Fake", "Real"],
                yticklabels=["Fake", "Real"],
                ax=ax,
            )
            ax.set_xlabel("Predicted", fontsize=11)
            ax.set_ylabel("Actual", fontsize=11)
            ax.set_title("Confusion Matrix — SilentWitness", fontsize=13)

            plt.tight_layout()
            plt.savefig(self.output_dir / "confusion_matrix.png", dpi=300)
            plt.close()
            logger.info("Confusion matrix saved")

        except ImportError:
            logger.warning("matplotlib/seaborn required for plot generation")

    def _write_markdown_report(self, metrics: dict) -> None:
        """Write a Markdown summary report."""
        report_path = self.output_dir / "evaluation_report.md"
        lines = [
            "# SilentWitness — Evaluation Report\n",
            f"**Total videos evaluated:** {metrics.get('total_videos', 0)}\n",
            "## Overall Metrics\n",
            "| Metric | Value |",
            "|---|---|",
            f"| AUC | {metrics.get('auc', 0):.4f} |",
            f"| Accuracy | {metrics.get('accuracy', 0):.4f} |",
            f"| Precision | {metrics.get('precision', 0):.4f} |",
            f"| Recall | {metrics.get('recall', 0):.4f} |",
            f"| F1 Score | {metrics.get('f1', 0):.4f} |",
            "",
            "## Per-Method Accuracy\n",
            "| Method | Accuracy | Avg Score | Count |",
            "|---|---|---|---|",
        ]

        for method, data in metrics.get("per_method", {}).items():
            lines.append(
                f"| {method} | {data['accuracy']:.4f} | "
                f"{data['avg_score']:.4f} | {data['count']} |"
            )

        lines.extend([
            "",
            "## Figures\n",
            "- ![ROC Curve](roc_curve.png)",
            "- ![Confusion Matrix](confusion_matrix.png)",
        ])

        report_path.write_text("\n".join(lines))
        logger.info(f"Report saved to {report_path}")
