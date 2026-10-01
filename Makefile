PYTHON ?= .venv/bin/python
UNIMATE_REFERENCE_DIR ?= $(abspath ../UniMate)

# An old environment value should not hide a usable local checkout. A path
# supplied on the make command line remains authoritative.
ifeq ($(origin UNIMATE_REFERENCE_DIR),command line)
REFERENCE_CANDIDATES := $(UNIMATE_REFERENCE_DIR)
else
REFERENCE_CANDIDATES := $(UNIMATE_REFERENCE_DIR) $(abspath ../UniMate) /private/tmp/unimate-reference
endif
REFERENCE_DIR := $(firstword $(foreach candidate,$(REFERENCE_CANDIDATES),$(if $(wildcard $(candidate)/data_process/mesh_animation/preprocess_char.py),$(candidate))))

.PHONY: studio studio-build studio-test

# Start the local API and Vite development server; Ctrl-C stops both.
studio:
	@test -n "$(REFERENCE_DIR)" || { echo "UniMate source not found. Pass UNIMATE_REFERENCE_DIR=/path/to/UniMate to make." >&2; exit 1; }
	@test -x "$(PYTHON)" || { echo "Python environment missing at $(PYTHON). See README setup." >&2; exit 1; }
	@test -d web/node_modules || { echo "Frontend dependencies missing. Run npm --prefix web install." >&2; exit 1; }
	@echo "Using UniMate source: $(REFERENCE_DIR)"
	@set -e; \
	UNIMATE_REFERENCE_DIR="$(REFERENCE_DIR)" "$(PYTHON)" -m unimate_mlx.studio_api & api_pid=$$!; \
	trap 'kill $$api_pid 2>/dev/null || true; wait $$api_pid 2>/dev/null || true' EXIT INT TERM; \
	npm --prefix web run dev

studio-build:
	npm --prefix web run build

studio-test:
	npm --prefix web test
	$(PYTHON) -m pytest -q tests/test_studio_api.py tests/test_sequence.py
