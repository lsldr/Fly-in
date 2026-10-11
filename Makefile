.PHONY: install run gui debug benchmark clean lint lint-strict

install:
	@uv sync

run:
	@uv run python -m fly_in $(ARGS)

debug:
	@uv run python -m pdb -m fly_in $(ARGS)

benchmark:
	@uv run python scripts/benchmark_ladder.py $(ARGS)

# Install micromamba if needed: "${SHELL}" <(curl -L micro.mamba.pm/install.sh)
QT_EXTRA_LIB := $(HOME)/.local/lib/qt-extra

# Creates a link file to the xcb-util-cursor library in a separate micromamba 
# environment, so that it can be used by PyQt6.
setup-gui-libs:
	micromamba create -y -n gui-libs -c conda-forge xcb-util-cursor
	mkdir -p $(QT_EXTRA_LIB)
	ln -sf $(HOME)/micromamba/envs/gui-libs/lib/libxcb-cursor.so* $(QT_EXTRA_LIB)/

gui:
	LD_LIBRARY_PATH="$(QT_EXTRA_LIB)$${LD_LIBRARY_PATH:+:$$LD_LIBRARY_PATH}" uv run fly-in-gui

lint:
	@uv run flake8 .
	@uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	@uv run flake8 .
	@uv run mypy . --strict

clean:
	@find . \
		-type d \( -name .git -o -name .venv -o -name venv \) -prune -o \
		-type d \( -name "__pycache__" -o -name .pytest_cache -o -name .mypy_cache \
			-o -name .ruff_cache -o -name .hypothesis -o -name .cache -o -name .tox \
			-o -name .nox -o -name .pytype \) -prune -exec rm -rf {} +
	@find . \
		-type d \( -name .git -o -name .venv -o -name venv \) -prune -o \
		-type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*.pyd" \
			-o -name ".dmypy.json" -o -name ".coverage" -o -name ".coverage.*" \
		\) -exec rm -f {} +
	@rm -rf build dist htmlcov
