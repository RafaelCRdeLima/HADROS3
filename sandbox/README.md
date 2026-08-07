# HADROS Sandbox

Material didático que explica **um conceito de cada vez** da física implementada
no [HADROS3](../README.md): notebooks curtos e independentes, e apostilas em PDF
para o que não cabe num notebook.

O HADROS3 é um pipeline grande: fonte UHE → geodésicas de Kerr → amostragem DIS →
POWHEG/PYTHIA → transporte Geant4 → imagem do observador. Cada estágio é correto e
testado, mas a cadeia inteira é difícil de auditar de cabeça, e impossível de usar
como material de introdução. Este repositório existe para resolver isso.

## Regras do sandbox

Todo notebook aqui obedece às mesmas regras. Se um notebook violar alguma delas,
ele deve ser dividido.

1. **Um conceito por notebook.** Se você precisa de duas frases com "e também"
   para descrever o que ele faz, são dois notebooks.
2. **Autocontido.** Nenhum `import hadros3`. As fórmulas são reimplementadas em
   poucas linhas, do zero, com comentários apontando para o arquivo e a linha do
   HADROS3 de onde vieram:
   ```python
   # HADROS3: hadros3/dis_sampler.py:1322
   d_tau = n_baryon * sigma * comoving_length_rg * r_g_cm
   ```
   Assim o notebook nunca quebra quando o HADROS3 muda — mas você sempre sabe onde
   conferir.
3. **A integral explicada.** Nada de fórmula caindo do céu: de onde vem, o que
   significa cada fator, quais são as unidades, e qual hipótese está escondida.
4. **Toda conta tem uma verificação.** Um limite analítico, uma fórmula fechada, um
   `assert`, um número conhecido da literatura. Um resultado numérico que ninguém
   pode conferir não vale nada.
5. **Roda em segundos** num laptop, só com `numpy` e `matplotlib`.
6. **Termina com exercícios** — incluindo pelo menos um cuja resposta não está no
   notebook.

## Apostilas

Textos longos, para ler antes ou junto com os notebooks. Compiladas em PDF e
versionadas prontas — não é preciso ter LaTeX instalado para ler.

| # | Apostila | Assunto |
|---|----------|---------|
| 01 | [dis.pdf](apostila/dis.pdf) — 33 páginas | Espalhamento profundamente inelástico: cinemática, tensores leptônico e hadrônico, modelo de pártons, DGLAP, corrente carregada e neutrinos UHE |

A apostila 01 traz as deduções completas (traços de Dirac, decomposição do tensor
hadrônico, Callan–Gross, jacobiano), figuras feitas com **dados reais** do HERA, e
38 referências — clássicas, modernas e didáticas — todas obtidas da API do
INSPIRE-HEP, nenhuma digitada à mão.

Para recompilar:

```bash
cd sandbox/apostila
make            # só o PDF
make tudo       # rebaixa os dados do HERA, refaz as figuras e compila
```

## Notebooks

| # | Notebook | Conceito |
|---|----------|----------|
| 01 | [01_profundidade_optica_dis.ipynb](notebooks/01_profundidade_optica_dis.ipynb) | Profundidade óptica $\tau = \int n_b \sigma\,\mathrm{d}\ell$ e a seção de choque DIS $\sigma_{\nu N}(E)$ |

### Planejados

| # | Conceito |
|---|----------|
| 02 | Onde a interação acontece: amostragem por CDF inversa, $P(i)=\Delta\tau_i/\sum_j\Delta\tau_j$ |
| 03 | A métrica de Kerr: horizonte, ergosfera, ISCO, observador ZAMO |
| 04 | Geodésicas nulas: integração, parâmetro afim, desvio para o vermelho gravitacional |
| 05 | Cinemática DIS: $x$, $y$, $Q^2$, $W$ e o estado final hadrônico |
| 06 | Da geodésica à imagem: o plano do observador e o *ray tracing* reverso |

## Como rodar

```bash
cd ~/Codes/HADROS3/sandbox
pip install -r requirements.txt
jupyter lab notebooks/
```

Os notebooks localizam `data/` subindo a árvore de diretórios, então funcionam
tanto abertos de dentro de `notebooks/` quanto da raiz do repositório.

## Dados

### `data/hera/` — dados públicos do HERA

Subconjuntos compactos extraídos da combinação H1+ZEUS
([Eur. Phys. J. C **75** (2015) 580](https://arxiv.org/abs/1506.06042)),
baixados de <https://www.desy.de/h1zeus/herapdf20/> por
[`apostila/baixar_dados_hera.py`](apostila/baixar_dados_hera.py):

| Arquivo | Conteúdo |
|---------|----------|
| `hera_nc_ep_920.dat` | 485 seções de choque reduzidas, corrente neutra $e^+p$ |
| `hera_cc_ep.dat` | 39 pontos de corrente carregada $e^+p$ |
| `herapdf20_lo_xf.dat` | PDFs HERAPDF2.0 LO em $Q^2 = 10$, $100$ e $10^4$ GeV² |

O extrator confere as regras de contagem de valência como teste de sanidade
($\int u_v\,dx = 2{,}0008$, $\int d_v\,dx = 1{,}0003$).

### `data/sigma/` — tabelas do HADROS3

`sandbox/data/sigma/sigma_nuN_CC_{GBW,IIM}.dat` são cópias literais de
[`data/sigma/`](../data/sigma) na raiz do HADROS3 — cópias de propósito, para que o
sandbox continue rodando mesmo se as tabelas de produção mudarem. Colunas:
`E_GeV`, `sigma_GeV^-2`, `sigma_cm2`; 300 pontos de $10^3$ a $10^{14}$ GeV.

> ⚠️ Estas tabelas ficam **acima** da parametrização de referência
> Gandhi–Quigg–Reno–Sarcevic (1998) por fatores de ~7 a ~90, dependendo da energia
> e do modelo. O notebook 01 §2.2 e a apostila 01 §6.4 mostram a comparação. Isso
> ainda **não** foi explicado — é o exercício 6 do notebook 01, o exercício 7 da
> apostila, e uma pendência real do HADROS3.

## Para os alunos

Leia a apostila 01 até o capítulo 4, faça o notebook 01, e então volte para os
capítulos 5 e 6 da apostila. Faça os exercícios antes de olhar o código de produção.
Depois abra [`hadros3/dis_sampler.py`](../hadros3/dis_sampler.py) e ache, no código
real, cada fórmula que você acabou de implementar — as referências
`arquivo.py:linha` nos comentários dos notebooks são o seu mapa. Quando as duas
coisas se encaixarem na sua cabeça, o notebook cumpriu a função dele.

Leitura de apoio: [`docs/Theory/HADROS3_Physics_Theory.tex`](../docs/Theory/HADROS3_Physics_Theory.tex).
