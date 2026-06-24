PYTHON ?= python3
HOST ?= 127.0.0.1
PORT ?= 8877

.PHONY: help hadros-web render-hadros-web render-camera-preview launch-camera-preview sample-uhe-source serve-hadros-web check clean

help:
	@echo "HADROS3 commands:"
	@echo "  make hadros-web        Serve the HADROS3 web control dashboard"
	@echo "  make render-hadros-web Render the HADROS3 geometry/configuration preview and exit"
	@echo "  make render-camera-preview Render only the HADROS3 camera preview"
	@echo "  make launch-camera-preview Open the original HADROS interactive camera preview"
	@echo "  make sample-uhe-source Generate H3-W5 UHE source samples through hadros-web"
	@echo "  make serve-hadros-web  Alias for make hadros-web"
	@echo "  make check             Run Python syntax checks"
	@echo "  make clean             Remove generated previews and Python caches"
	@echo ""
	@echo "Variables:"
	@echo "  PYTHON=$(PYTHON)"
	@echo "  HOST=$(HOST)"
	@echo "  PORT=$(PORT)"

hadros-web:
	$(PYTHON) hadros_web.py --serve --host $(HOST) --port $(PORT)

render-hadros-web:
	$(PYTHON) hadros_web.py

render-camera-preview:
	$(PYTHON) hadros_web.py --camera-preview-only

launch-camera-preview:
	$(PYTHON) hadros_web.py --launch-interactive-camera

sample-uhe-source:
	$(PYTHON) hadros_web.py --sample-uhe-source

serve-hadros-web:
	$(MAKE) hadros-web

check:
	$(PYTHON) -m py_compile hadros_web.py hadros3/*.py tests/test_hadros_web.py tests/test_uhe_source.py

clean:
	rm -rf output
	rm -rf __pycache__ hadros3/__pycache__ tests/__pycache__ .pytest_cache
