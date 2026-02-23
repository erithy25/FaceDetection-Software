# SilentWitness — Build & Training Pipeline
#
# Usage:
#   make setup          Install all dependencies (Node + Python)
#   make preprocess     Extract faces from FaceForensics++ dataset
#   make train          Train EfficientNet-B0 + LSTM (full pipeline)
#   make export         Export to ONNX + INT8 quantization
#   make run-backend    Start Python backend standalone (dev mode)
#   make dev            Start full Tauri dev environment
#   make pipeline       Run the complete ML pipeline end-to-end
#
# Variables (override on command line):
#   DATASET_ROOT  Path to FaceForensics++ dataset
#   DATA_DIR      Path to preprocessed face crops (default: data/processed)
#   MODEL_DIR     Path to trained model checkpoints (default: python/models)
#   PYTHON        Python interpreter (default: auto-detect venv or system)

# ─── Configuration ──────────────────────────────────────────────────

DATASET_ROOT ?= data/FaceForensics++
DATA_DIR     ?= data/processed
MODEL_DIR    ?= python/models
EPOCHS_EFF   ?= 30
EPOCHS_LSTM  ?= 50
BATCH_SIZE   ?= 32

# Auto-detect Python
PYTHON := $(shell \
	if [ -f .venv/bin/python ]; then echo .venv/bin/python; \
	elif [ -f venv/bin/python ]; then echo venv/bin/python; \
	elif command -v python3 > /dev/null 2>&1; then echo python3; \
	else echo python; fi)

# ─── Setup ──────────────────────────────────────────────────────────

.PHONY: setup setup-python setup-node

setup: setup-python setup-node
	@echo "✓ All dependencies installed"

setup-python:
	@echo "→ Creating Python virtual environment..."
	$(PYTHON) -m venv .venv || true
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r python/requirements.txt
	@echo "✓ Python dependencies installed"

setup-node:
	@echo "→ Installing Node dependencies..."
	npm install
	@echo "✓ Node dependencies installed"

# ─── ML Pipeline ────────────────────────────────────────────────────

.PHONY: preprocess train train-efficientnet train-lstm export pipeline

preprocess:
	@echo "→ Preprocessing FaceForensics++ dataset..."
	@echo "  Dataset root: $(DATASET_ROOT)"
	@echo "  Output dir:   $(DATA_DIR)"
	$(PYTHON) python/preprocess_dataset.py \
		--dataset_root "$(DATASET_ROOT)" \
		--output_dir "$(DATA_DIR)" \
		--compression c23 \
		--max_frames_per_video 50
	@echo "✓ Preprocessing complete"

train-efficientnet:
	@echo "→ Training EfficientNet-B0..."
	$(PYTHON) python/train_efficientnet.py \
		--data_dir "$(DATA_DIR)" \
		--output_dir "$(MODEL_DIR)" \
		--epochs $(EPOCHS_EFF) \
		--batch_size $(BATCH_SIZE)
	@echo "✓ EfficientNet training complete"

train-lstm: train-efficientnet
	@echo "→ Training Temporal LSTM..."
	$(PYTHON) python/train_lstm.py \
		--data_dir "$(DATA_DIR)" \
		--efficientnet_model "$(MODEL_DIR)/best_efficientnet_b0_deepfake.pt" \
		--output_dir "$(MODEL_DIR)" \
		--epochs $(EPOCHS_LSTM) \
		--batch_size $(BATCH_SIZE)
	@echo "✓ LSTM training complete"

train: train-lstm
	@echo "✓ All models trained"

export:
	@echo "→ Exporting models to ONNX + INT8 quantization..."
	$(PYTHON) python/export_onnx.py \
		--efficientnet_model "$(MODEL_DIR)/best_efficientnet_b0_deepfake.pt" \
		--lstm_model "$(MODEL_DIR)/temporal_lstm_best.pt" \
		--calibration_dir "$(DATA_DIR)/train/real" \
		--output_dir "$(MODEL_DIR)"
	@echo "✓ ONNX export complete"

pipeline: preprocess train export
	@echo ""
	@echo "═══════════════════════════════════════════════"
	@echo "  SilentWitness ML Pipeline — COMPLETE"
	@echo "  Models ready in: $(MODEL_DIR)/"
	@echo "═══════════════════════════════════════════════"

# ─── Evaluation ─────────────────────────────────────────────────────

.PHONY: evaluate

evaluate:
	@echo "→ Running evaluation on FaceForensics++ test set..."
	$(PYTHON) -c "from python.evaluator import Evaluator; \
		e = Evaluator('$(DATASET_ROOT)', 'evaluation_results'); \
		m = e.evaluate_dataset(compression='c23'); \
		print(f\"AUC: {m.get('auc', 0):.4f}, Accuracy: {m.get('accuracy', 0):.4f}\")"
	@echo "✓ Evaluation complete — results in evaluation_results/"

# ─── Development ────────────────────────────────────────────────────

.PHONY: run-backend dev build

run-backend:
	@echo "→ Starting Python backend (standalone dev mode)..."
	$(PYTHON) python/main.py --debug

dev:
	@echo "→ Starting Tauri dev environment..."
	npm run tauri dev

build:
	@echo "→ Building production Tauri app..."
	npm run tauri build

# ─── Utility ────────────────────────────────────────────────────────

.PHONY: clean clean-models clean-data

clean:
	rm -rf dist/ target/ node_modules/ .venv/ __pycache__/
	find python -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleaned build artifacts"

clean-models:
	rm -f python/models/*.onnx python/models/*.pt
	@echo "✓ Cleaned model files"

clean-data:
	rm -rf $(DATA_DIR)
	@echo "✓ Cleaned preprocessed data"

.PHONY: help

help:
	@echo "SilentWitness — Available targets:"
	@echo ""
	@echo "  Setup:"
	@echo "    make setup            Install all dependencies"
	@echo "    make setup-python     Install Python deps only"
	@echo "    make setup-node       Install Node deps only"
	@echo ""
	@echo "  ML Pipeline:"
	@echo "    make preprocess       Extract faces from FF++ dataset"
	@echo "    make train            Train EfficientNet + LSTM"
	@echo "    make export           Export to ONNX + INT8"
	@echo "    make pipeline         Run complete ML pipeline"
	@echo "    make evaluate         Evaluate on FF++ test set"
	@echo ""
	@echo "  Development:"
	@echo "    make run-backend      Start Python backend (dev mode)"
	@echo "    make dev              Start Tauri dev environment"
	@echo "    make build            Build production app"
	@echo ""
	@echo "  Cleanup:"
	@echo "    make clean            Remove build artifacts"
	@echo "    make clean-models     Remove trained models"
	@echo "    make clean-data       Remove preprocessed data"
	@echo ""
	@echo "  Variables:"
	@echo "    DATASET_ROOT=$(DATASET_ROOT)"
	@echo "    DATA_DIR=$(DATA_DIR)"
	@echo "    MODEL_DIR=$(MODEL_DIR)"
