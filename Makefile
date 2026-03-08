.PHONY: install build clean-cache test reproduce

# Default data directory (override with: make reproduce DATA_DIR=/path/to/OpenML-CC18)
DATA_DIR ?= $(HOME)/pcmabinf/OpenML-CC18

# Reproduce settings matching the original notebook
SIMULATIONS       ?= 64
BATCH_COUNT       ?= 100
BATCH_SIZE        ?= 100
REWARD_VARIANCE   ?= 1.0
EPSILON           ?= 0.01
N_JOBS            ?= -1
OUTPUT_DIR        ?= results
TASK_MAX_FEATURES ?= 100

install:
	uv sync --group dev

build:
	uv build

clean-cache:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache  -exec rm -rf {} +
	find . -type d -name dist         -exec rm -rf {} +
	find . -type d -name '*.egg-info' -exec rm -rf {} +

test:
	uv run pytest --tb=short

reproduce:
	uv run pcmabinf run \
		--data-dir           "$(DATA_DIR)" \
		--simulations        $(SIMULATIONS) \
		--batch-count        $(BATCH_COUNT) \
		--batch-size         $(BATCH_SIZE) \
		--reward-variance    $(REWARD_VARIANCE) \
		--epsilon-multiplier $(EPSILON) \
		--n-jobs             $(N_JOBS) \
		--task-max-features  $(TASK_MAX_FEATURES) \
		--output-dir         $(OUTPUT_DIR)
