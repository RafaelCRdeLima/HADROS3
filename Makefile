PYTHON ?= python3
HOST ?= 127.0.0.1
PORT ?= 8877

CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -pedantic
CPP_INCLUDES := -Icpp/include
KERR_PORT_SRC := cpp/src/kerr/kerr_metric.cpp cpp/src/kerr/kerr_geodesic.cpp cpp/src/cascade/kerr_local_tetrad.cpp cpp/src/cascade/packet_kerr_null_propagator.cpp

.PHONY: help cpp hadros3-forward-geodesics hadros-web render-hadros-web render-camera-preview launch-camera-preview sample-uhe-source propagate-forward-geodesics sample-dis-interactions serve-hadros-web check clean

help:
	@echo "HADROS3 commands:"
	@echo "  make hadros-web        Serve the HADROS3 web control dashboard"
	@echo "  make render-hadros-web Render the HADROS3 geometry/configuration preview and exit"
	@echo "  make render-camera-preview Render only the HADROS3 camera preview"
	@echo "  make launch-camera-preview Open the original HADROS interactive camera preview"
	@echo "  make sample-uhe-source Generate H3-W5 UHE source samples through hadros-web"
	@echo "  make cpp               Build HADROS3 C++ physics backends"
	@echo "  make propagate-forward-geodesics Generate H3-W6 forward geodesics through hadros-web"
	@echo "  make sample-dis-interactions Generate H3-W7 DIS interaction samples through hadros-web"
	@echo "  make serve-hadros-web  Alias for make hadros-web"
	@echo "  make check             Run Python syntax checks"
	@echo "  make clean             Remove generated previews and Python caches"
	@echo ""
	@echo "Variables:"
	@echo "  PYTHON=$(PYTHON)"
	@echo "  HOST=$(HOST)"
	@echo "  PORT=$(PORT)"

cpp: bin/hadros3_forward_geodesics

hadros3-forward-geodesics: bin/hadros3_forward_geodesics

bin/hadros3_forward_geodesics: cpp/apps/hadros3_forward_geodesics.cpp $(KERR_PORT_SRC) cpp/include/geodesic_state.hpp cpp/include/kerr_metric.hpp cpp/include/kerr_metric_derivatives.hpp cpp/include/kerr_geodesic.hpp cpp/include/hadros/cascade/kerr_local_tetrad.hpp cpp/include/hadros/cascade/packet_kerr_null_propagator.hpp cpp/include/hadros/cascade/types.hpp
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $(CPP_INCLUDES) cpp/apps/hadros3_forward_geodesics.cpp $(KERR_PORT_SRC) -o $@

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
	$(PYTHON) -m py_compile hadros_web.py hadros3/*.py tests/test_hadros_web.py tests/test_uhe_source.py tests/test_forward_geodesics.py tests/test_dis_sampler.py

clean:
	rm -rf output
	rm -rf __pycache__ hadros3/__pycache__ tests/__pycache__ .pytest_cache
