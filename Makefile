PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
HOST ?= 127.0.0.1
PORT ?= 8877

CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -pedantic
NVCC_CANDIDATE := $(shell command -v nvcc 2>/dev/null)
NVCC ?= $(if $(NVCC_CANDIDATE),$(NVCC_CANDIDATE),nvcc)
NVCCFLAGS ?= -O3 -std=c++17
CPP_INCLUDES := -Icpp/include
KERR_PORT_SRC := cpp/src/kerr/kerr_metric.cpp cpp/src/kerr/kerr_geodesic.cpp cpp/src/cascade/kerr_local_tetrad.cpp cpp/src/cascade/packet_kerr_null_propagator.cpp

.PHONY: help install-dev test cpp hadros3-forward-geodesics hadros3-dis-sampler hadros3-geodesic-preview-cuda hadros-web render-hadros-web render-camera-preview launch-camera-preview sample-uhe-source propagate-forward-geodesics sample-dis-interactions serve-hadros-web check validate clean

help:
	@echo "HADROS3 commands:"
	@echo "  make install-dev       Install development dependencies"
	@echo "  make test              Run the Python test suite"
	@echo "  make hadros-web        Serve the HADROS3 web control dashboard"
	@echo "  make render-hadros-web Render the HADROS3 geometry/configuration preview and exit"
	@echo "  make render-camera-preview Render only the HADROS3 camera preview"
	@echo "  make launch-camera-preview Open the original HADROS interactive camera preview"
	@echo "  make sample-uhe-source Generate H3-W5 UHE source samples through hadros-web"
	@echo "  make cpp               Build HADROS3 C++ physics backends"
	@echo "  make hadros3-dis-sampler Build the self-contained H3-W7 C++ DIS sampler"
	@echo "  make hadros3-geodesic-preview-cuda Build self-contained HADROS3 CUDA camera preview if CUDA is available"
	@echo "  make propagate-forward-geodesics Generate H3-W6 forward geodesics through hadros-web"
	@echo "  make sample-dis-interactions Generate H3-W7 DIS interaction samples through hadros-web"
	@echo "  make serve-hadros-web  Alias for make hadros-web"
	@echo "  make check             Run syntax checks and the Python test suite"
	@echo "  make validate          Build C++ backends and run full checks"
	@echo "  make clean             Remove generated previews and Python caches"
	@echo ""
	@echo "Variables:"
	@echo "  PYTHON=$(PYTHON)"
	@echo "  PIP=$(PIP)"
	@echo "  HOST=$(HOST)"
	@echo "  PORT=$(PORT)"

install-dev:
	$(PIP) install -r requirements-dev.txt

test:
	$(PYTHON) -m pytest tests

cpp: bin/hadros3_forward_geodesics bin/hadros3_dis_sampler

hadros3-forward-geodesics: bin/hadros3_forward_geodesics

hadros3-dis-sampler: bin/hadros3_dis_sampler

hadros3-geodesic-preview-cuda:
	@mkdir -p bin
	@if command -v $(NVCC) >/dev/null 2>&1; then \
	  echo "[hadros3_geodesic_preview_cuda] Building self-contained CUDA preview renderer"; \
	  if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists glfw3; then \
	    $(NVCC) $(NVCCFLAGS) -DHADROS_CUDA_PREVIEW_GLFW cpp/cuda/hadros3_geodesic_preview_cuda.cu -o bin/hadros3_geodesic_preview_cuda $$(pkg-config --cflags --libs glfw3) -lGL; \
	  else \
	    echo "[hadros3_geodesic_preview_cuda] GLFW not found; building headless CUDA preview renderer"; \
	    $(NVCC) $(NVCCFLAGS) cpp/cuda/hadros3_geodesic_preview_cuda.cu -o bin/hadros3_geodesic_preview_cuda; \
	  fi; \
	else \
	  echo "[hadros3_geodesic_preview_cuda] nvcc not found: $(NVCC)"; \
	  echo "[hadros3_geodesic_preview_cuda] HADROS3 CUDA preview unavailable; camera preview will use fallback."; \
	fi

bin/hadros3_forward_geodesics: cpp/apps/hadros3_forward_geodesics.cpp $(KERR_PORT_SRC) cpp/include/geodesic_state.hpp cpp/include/kerr_metric.hpp cpp/include/kerr_metric_derivatives.hpp cpp/include/kerr_geodesic.hpp cpp/include/hadros/cascade/kerr_local_tetrad.hpp cpp/include/hadros/cascade/packet_kerr_null_propagator.hpp cpp/include/hadros/cascade/types.hpp
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $(CPP_INCLUDES) cpp/apps/hadros3_forward_geodesics.cpp $(KERR_PORT_SRC) -o $@

bin/hadros3_dis_sampler: cpp/apps/hadros3_dis_sampler.cpp
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $(CPP_INCLUDES) cpp/apps/hadros3_dis_sampler.cpp -o $@

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

propagate-forward-geodesics:
	$(PYTHON) hadros_web.py --propagate-forward-geodesics

sample-dis-interactions:
	$(PYTHON) hadros_web.py --sample-dis-interactions

serve-hadros-web:
	$(MAKE) hadros-web

check:
	$(PYTHON) -m py_compile hadros_web.py hadros3/*.py
	$(PYTHON) -m pytest tests

validate:
	$(MAKE) cpp
	$(MAKE) check

clean:
	rm -rf output
	rm -rf __pycache__ hadros3/__pycache__ tests/__pycache__ .pytest_cache
