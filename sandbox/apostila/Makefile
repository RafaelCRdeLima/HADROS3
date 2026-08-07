# Apostila de DIS do HADROS Sandbox.
#
#   make            compila dis.pdf (assume figuras e dados já presentes)
#   make figuras    regenera as figuras a partir dos dados em ../data/
#   make dados      rebaixa os dados públicos do HERA (precisa de rede, ~28 MB)
#   make referencias  regenera dis.bib a partir do INSPIRE-HEP (precisa de rede)
#   make tudo       dados + figuras + pdf
#   make limpar     remove os arquivos auxiliares do LaTeX
#   make limpar-tudo  idem, e também dis.pdf

FIGURAS := $(wildcard figuras/*.pdf)

.PHONY: all tudo figuras dados referencias limpar limpar-tudo

all: dis.pdf

dis.pdf: dis.tex dis.bib $(FIGURAS)
	latexmk -pdf -interaction=nonstopmode dis.tex

figuras:
	cd figuras && python3 gerar_figuras.py

dados:
	python3 baixar_dados_hera.py

referencias:
	python3 buscar_referencias.py

tudo: dados figuras dis.pdf

limpar:
	latexmk -c dis.tex

limpar-tudo:
	latexmk -C dis.tex
