# ---------------------------------------------------------------------------
# Environment resolution
#
# `make setup` writes .hadros3-env.mk with the interpreter and the library
# prefixes resolved for THIS machine (it is generated, gitignored, and never
# committed). Everything below is only a best-effort fallback for a checkout
# where setup has not been run yet, so that no path is hardcoded to a single
# developer's machine.
# ---------------------------------------------------------------------------
-include .hadros3-env.mk

BOOTSTRAP_PYTHON ?= python3
MICROMAMBA := $(shell command -v micromamba 2>/dev/null)
CONDA_ENV_GUESS := $(firstword $(wildcard $(HOME)/micromamba/envs/hadros3 $(HOME)/micromamba/envs/dis) $(CONDA_PREFIX))
CONDA_ENV_NAME := $(notdir $(CONDA_ENV_GUESS))
VENV_PYTHON := $(wildcard $(CURDIR)/.venv/bin/python)
PYTHON_FALLBACK := $(if $(VENV_PYTHON),$(VENV_PYTHON),$(BOOTSTRAP_PYTHON))

PYTHON ?= $(if $(and $(MICROMAMBA),$(CONDA_ENV_NAME)),$(MICROMAMBA) run -n $(CONDA_ENV_NAME) python,$(PYTHON_FALLBACK))
PIP ?= $(PYTHON) -m pip
HOST ?= 127.0.0.1
PORT ?= 8877

CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -pedantic
NVCC_CANDIDATE := $(shell command -v nvcc 2>/dev/null)
NVCC ?= $(if $(NVCC_CANDIDATE),$(NVCC_CANDIDATE),nvcc)
NVCCFLAGS ?= -O3 -std=c++17
# Emit code for every major architecture supported by the local toolkit, so a
# binary built on one machine can also run on a different NVIDIA GPU.
# Override with NVCC_ARCH_FLAGS= for CUDA toolkits older than 11.5.
NVCC_ARCH_FLAGS ?= -arch=all-major
CPP_INCLUDES := -Icpp/include
KERR_PORT_SRC := cpp/src/kerr/kerr_metric.cpp cpp/src/kerr/kerr_geodesic.cpp cpp/src/cascade/kerr_local_tetrad.cpp cpp/src/cascade/packet_kerr_null_propagator.cpp

.PHONY: setup setup-light doctor all print-env cpp-core cpp-optional cpp-all hadros3-geodesic-preview-cuda-optional
.PHONY: help install-dev test cpp hadros3-forward-geodesics hadros3-dis-sampler hadros3-observer-bridge hadros3-powheg-driver hadros3-event-generator hadros3-geant4-transport geant4-build geant4-environment-check geant4-import-check geant4-vacuum-smoke geant4-material-smoke geant4-real-free geant4-validate hadros3-geodesic-preview-cuda powheg-fetch powheg-build powheg-smoke powheg powheg-real-smoke powheg-real-free event-generation-dry-run event-generation-parton-check event-generation-real-smoke event-generation-real-free hadros-web render-hadros-web render-camera-preview launch-camera-preview sample-uhe-source propagate-forward-geodesics sample-dis-interactions observer-bridge observer-image-branches serve-hadros-web release-software release-physics release-pipeline theory check validate clean

help:
	@echo "HADROS3 commands:"
	@echo "  make setup             Configure this machine (conda env if available, else .venv)"
	@echo "  make setup-light       Configure only the pure-Python layer in .venv (no conda)"
	@echo "  make doctor            Report what works on this machine and how to fix what does not"
	@echo "  make all               Build every backend the machine can actually build"
	@echo "  make cpp-core          Build the four backends that need nothing but g++"
	@echo "  make install-dev       Install development dependencies"
	@echo "  make test              Run the Python test suite"
	@echo "  make hadros-web        Serve the HADROS3 web control dashboard"
	@echo "  make render-hadros-web Render the HADROS3 geometry/configuration preview and exit"
	@echo "  make render-camera-preview Render only the HADROS3 camera preview"
	@echo "  make launch-camera-preview Open the original HADROS interactive camera preview"
	@echo "  make sample-uhe-source Generate H3-W5 UHE source samples through hadros-web"
	@echo "  make cpp               Build HADROS3 C++ physics backends"
	@echo "  make hadros3-dis-sampler Build the self-contained H3-W7 C++ DIS sampler"
	@echo "  make hadros3-observer-bridge Build the self-contained H3-W8 C++ Observer Bridge scorer"
	@echo "  make hadros3-powheg-driver Build the self-contained H3-W9a C++ POWHEG dry-run driver"
	@echo "  make hadros3-geodesic-preview-cuda Build self-contained HADROS3 CUDA camera preview if CUDA is available"
	@echo "  make powheg-fetch     Fetch/copy the pinned POWHEG-BOX-RES DIS source into external/powheg"
	@echo "  make powheg-build     Build local POWHEG DIS pwhg_main for H3-W9 bootstrap"
	@echo "  make powheg-smoke     Run a minimal local POWHEG DIS smoke test"
	@echo "  make powheg           Prepare H3-W9a POWHEG dry-run jobs through hadros-web"
	@echo "  make powheg-real-smoke Run H3-W9b one-candidate local POWHEG LHE smoke mode"
	@echo "  make powheg-real-free Run H3-W9b local POWHEG with configured candidate/event counts"
	@echo "  make hadros3-event-generator Build the H3-W10 PYTHIA 8/HepMC3 backend"
	@echo "  make event-generation-real-smoke Run H3-W10 on at most two LHE events"
	@echo "  make geant4-build      Build the H3-W11 Geant4/HepMC3 backend"
	@echo "  make geant4-import-check Audit H3-W10 input and the supported physics domain"
	@echo "  make geant4-vacuum-smoke Run an explicit H3-W11 vacuum smoke"
	@echo "  make geant4-material-smoke Run an explicit H3-W11 local-material smoke"
	@echo "  make geant4-validate   Run focused H3-W11 numerical tests"
	@echo "  make propagate-forward-geodesics Generate H3-W6 forward geodesics through hadros-web"
	@echo "  make sample-dis-interactions Generate H3-W7 DIS interaction samples through hadros-web"
	@echo "  make observer-bridge   Generate H3-W8 Observer Bridge scoring products through hadros-web"
	@echo "  make observer-image-branches Generate H3-W8b Observer Image Branch Analysis products"
	@echo "  make release-software  Increment software_version and rebuild the Theory PDF"
	@echo "  make release-physics   Increment physics_version/theory_version and rebuild the Theory PDF"
	@echo "  make release-pipeline PIPELINE=H3-W10 Update pipeline_version and rebuild the Theory PDF"
	@echo "  make theory            Rebuild docs/Theory/HADROS3_Physics_Theory.pdf"
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

setup:
	$(BOOTSTRAP_PYTHON) scripts/setup/bootstrap_environment.py $(ARGS)

setup-light:
	$(BOOTSTRAP_PYTHON) scripts/setup/bootstrap_environment.py --light $(ARGS)

doctor:
	@$(BOOTSTRAP_PYTHON) scripts/setup/doctor.py $(ARGS)

print-env:
	@echo "PYTHON=$(PYTHON)"
	@echo "PYTHIA8_PREFIX=$(PYTHIA8_PREFIX)"
	@echo "GEANT4_PREFIX=$(GEANT4_PREFIX)"
	@echo "NVCC=$(NVCC)"

# Builds everything this machine is actually able to build, and reports what it
# had to skip, instead of failing on the first unavailable dependency.
all: cpp-core cpp-optional hadros3-geodesic-preview-cuda-optional
	@echo ""
	@echo "[all] done -- run 'make doctor' for the full capability report"

install-dev:
	$(PIP) install -r requirements-dev.txt

test:
	$(PYTHON) -m pytest tests

# `cpp` stays the friendly target: core backends always, optional ones when
# their dependencies are present. `cpp-all` is the strict variant.
cpp: cpp-core cpp-optional

cpp-core: bin/hadros3_forward_geodesics bin/hadros3_dis_sampler bin/hadros3_observer_bridge bin/hadros3_powheg_driver

cpp-optional:
	@if [ -f "$(PYTHIA8_PREFIX)/include/Pythia8/Pythia.h" ]; then \
	  $(MAKE) --no-print-directory bin/hadros3_event_generator; \
	else \
	  echo "[skip] PYTHIA 8 not found under PYTHIA8_PREFIX='$(PYTHIA8_PREFIX)'"; \
	  echo "[skip] hadros3_event_generator not built -- run 'make setup' with micromamba/conda installed"; \
	fi
	@if [ -d "$(GEANT4_PREFIX)/lib/cmake/Geant4" ] || [ -d "$(GEANT4_PREFIX)/lib64/cmake/Geant4" ]; then \
	  $(MAKE) --no-print-directory bin/hadros3_geant4_transport; \
	else \
	  echo "[skip] Geant4 not found under GEANT4_PREFIX='$(GEANT4_PREFIX)'"; \
	  echo "[skip] hadros3_geant4_transport not built -- run 'make setup' with micromamba/conda installed"; \
	fi

cpp-all: cpp-core bin/hadros3_event_generator bin/hadros3_geant4_transport

hadros3-forward-geodesics: bin/hadros3_forward_geodesics

hadros3-dis-sampler: bin/hadros3_dis_sampler

hadros3-observer-bridge: bin/hadros3_observer_bridge

hadros3-powheg-driver: bin/hadros3_powheg_driver

hadros3-event-generator: bin/hadros3_event_generator

hadros3-geant4-transport geant4-build: bin/hadros3_geant4_transport

# Explicit request: a missing nvcc is an error, because the user asked for the
# CUDA preview by name. `hadros3-geodesic-preview-cuda-optional` is the variant
# used by `make all`, which only warns.
hadros3-geodesic-preview-cuda: HADROS3_CUDA_REQUIRED := 1
hadros3-geodesic-preview-cuda-optional: HADROS3_CUDA_REQUIRED := 0

hadros3-geodesic-preview-cuda hadros3-geodesic-preview-cuda-optional:
	@mkdir -p bin
	@if ! command -v $(NVCC) >/dev/null 2>&1; then \
	  echo "[hadros3_geodesic_preview_cuda] nvcc not found: $(NVCC)"; \
	  echo "[hadros3_geodesic_preview_cuda] the interactive camera preview needs the CUDA toolkit and an NVIDIA GPU."; \
	  echo "[hadros3_geodesic_preview_cuda] install it with: sudo apt install nvidia-cuda-toolkit libglfw3-dev pkg-config"; \
	  echo "[hadros3_geodesic_preview_cuda] then run 'make doctor' to confirm."; \
	  if [ "$(HADROS3_CUDA_REQUIRED)" = "1" ]; then exit 1; fi; \
	  exit 0; \
	fi; \
	if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists glfw3; then \
	  echo "[hadros3_geodesic_preview_cuda] Building self-contained CUDA preview renderer (interactive, GLFW)"; \
	  $(NVCC) $(NVCCFLAGS) $(NVCC_ARCH_FLAGS) -DHADROS_CUDA_PREVIEW_GLFW cpp/cuda/hadros3_geodesic_preview_cuda.cu -o bin/hadros3_geodesic_preview_cuda $$(pkg-config --cflags --libs glfw3) -lGL; \
	else \
	  echo "[hadros3_geodesic_preview_cuda] GLFW not found; building HEADLESS CUDA preview renderer (no interactive window)"; \
	  echo "[hadros3_geodesic_preview_cuda] install libglfw3-dev and rebuild to get the interactive window"; \
	  $(NVCC) $(NVCCFLAGS) $(NVCC_ARCH_FLAGS) cpp/cuda/hadros3_geodesic_preview_cuda.cu -o bin/hadros3_geodesic_preview_cuda; \
	fi

bin/hadros3_forward_geodesics: cpp/apps/hadros3_forward_geodesics.cpp $(KERR_PORT_SRC) cpp/include/geodesic_state.hpp cpp/include/kerr_metric.hpp cpp/include/kerr_metric_derivatives.hpp cpp/include/kerr_geodesic.hpp cpp/include/hadros/cascade/kerr_local_tetrad.hpp cpp/include/hadros/cascade/packet_kerr_null_propagator.hpp cpp/include/hadros/cascade/types.hpp
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $(CPP_INCLUDES) cpp/apps/hadros3_forward_geodesics.cpp $(KERR_PORT_SRC) -o $@

bin/hadros3_dis_sampler: cpp/apps/hadros3_dis_sampler.cpp
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $(CPP_INCLUDES) cpp/apps/hadros3_dis_sampler.cpp -o $@

bin/hadros3_observer_bridge: cpp/apps/hadros3_observer_bridge.cpp
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $(CPP_INCLUDES) cpp/apps/hadros3_observer_bridge.cpp -o $@

bin/hadros3_powheg_driver: cpp/apps/hadros3_powheg_driver.cpp
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $(CPP_INCLUDES) cpp/apps/hadros3_powheg_driver.cpp -o $@

PYTHIA8_PREFIX ?= $(CONDA_ENV_GUESS)
PYTHIA8_CXX ?= $(firstword $(wildcard $(PYTHIA8_PREFIX)/bin/*-linux-gnu-c++) $(CXX))

bin/hadros3_event_generator: cpp/apps/hadros3_event_generator.cpp
	@mkdir -p bin
	$(PYTHIA8_CXX) $(CXXFLAGS) -I$(PYTHIA8_PREFIX)/include $< -L$(PYTHIA8_PREFIX)/lib -Wl,-rpath,$(PYTHIA8_PREFIX)/lib -lpythia8 -lHepMC3 -lz -ldl -pthread -o $@

GEANT4_PREFIX ?= $(CONDA_ENV_GUESS)
GEANT4_CMAKE ?= $(firstword $(wildcard $(GEANT4_PREFIX)/bin/cmake) cmake)

bin/hadros3_geant4_transport: cpp/apps/hadros3_geant4_transport.cpp cpp/geant4/CMakeLists.txt
	$(GEANT4_CMAKE) -S cpp/geant4 -B build/geant4 -G Ninja -DCMAKE_PREFIX_PATH=$(GEANT4_PREFIX) -DCMAKE_BUILD_TYPE=Release
	$(GEANT4_CMAKE) --build build/geant4 --parallel 2
	@mkdir -p bin
	cp build/geant4/hadros3_geant4_transport $@

powheg-fetch:
	$(PYTHON) scripts/powheg/bootstrap_powheg.py fetch

powheg-build:
	$(PYTHON) scripts/powheg/bootstrap_powheg.py build

powheg-smoke:
	$(PYTHON) scripts/powheg/bootstrap_powheg.py smoke

powheg:
	$(PYTHON) hadros_web.py --powheg

powheg-real-smoke:
	$(PYTHON) hadros_web.py --powheg-real-smoke

powheg-real-free:
	$(PYTHON) hadros_web.py --powheg-real-free

event-generation-dry-run: bin/hadros3_event_generator
	$(PYTHON) hadros_web.py --event-generation-dry-run

event-generation-parton-check: bin/hadros3_event_generator
	$(PYTHON) hadros_web.py --event-generation-parton-check

event-generation-real-smoke: bin/hadros3_event_generator
	$(PYTHON) hadros_web.py --event-generation-real-smoke

event-generation-real-free: bin/hadros3_event_generator
	$(PYTHON) hadros_web.py --event-generation-real-free

geant4-environment-check: bin/hadros3_geant4_transport
	$(PYTHON) hadros_web.py --geant4-environment-check

geant4-import-check: bin/hadros3_geant4_transport
	$(PYTHON) hadros_web.py --geant4-import-check

geant4-vacuum-smoke: bin/hadros3_geant4_transport
	$(PYTHON) hadros_web.py --geant4-vacuum-smoke

geant4-material-smoke: bin/hadros3_geant4_transport
	$(PYTHON) hadros_web.py --geant4-material-smoke

geant4-real-free: bin/hadros3_geant4_transport
	$(PYTHON) hadros_web.py --geant4-real-free

geant4-validate: bin/hadros3_geant4_transport
	$(PYTHON) -m pytest tests/test_geant4_transport.py -v

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

observer-bridge:
	$(PYTHON) hadros_web.py --observer-bridge

observer-image-branches:
	$(PYTHON) hadros_web.py --observer-image-branches

serve-hadros-web:
	$(MAKE) hadros-web

release-software:
	$(PYTHON) scripts/release/update_version.py --software
	$(MAKE) theory

release-physics:
	$(PYTHON) scripts/release/update_version.py --physics
	$(MAKE) theory

release-pipeline:
	@if [ -z "$(PIPELINE)" ]; then echo "PIPELINE is required, for example: make release-pipeline PIPELINE=H3-W10"; exit 2; fi
	$(PYTHON) scripts/release/update_version.py --pipeline $(PIPELINE)
	$(MAKE) theory

theory:
	cd docs/Theory && pdflatex -interaction=nonstopmode HADROS3_Physics_Theory.tex
	cd docs/Theory && pdflatex -interaction=nonstopmode HADROS3_Physics_Theory.tex
	cd docs/Theory && pdflatex -interaction=nonstopmode HADROS3_Physics_Theory.tex
	rm -f docs/Theory/HADROS3_Physics_Theory.aux docs/Theory/HADROS3_Physics_Theory.out docs/Theory/HADROS3_Physics_Theory.toc docs/Theory/HADROS3_Physics_Theory.log

check:
	$(PYTHON) -m py_compile hadros_web.py hadros3/*.py
	$(PYTHON) -m pytest tests

validate:
	$(MAKE) cpp
	$(MAKE) check

clean:
	rm -rf output
	rm -rf __pycache__ hadros3/__pycache__ tests/__pycache__ .pytest_cache
