# HADROS Sandbox

Notebooks pequenos e independentes que explicam **um conceito de cada vez** da
física implementada no [HADROS3](../README.md).

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

`sandbox/data/sigma/sigma_nuN_CC_{GBW,IIM}.dat` são cópias literais de
[`data/sigma/`](../data/sigma) na raiz do HADROS3 — cópias de propósito, para que o
sandbox continue rodando mesmo se as tabelas de produção mudarem. Colunas:
`E_GeV`, `sigma_GeV^-2`, `sigma_cm2`; 300 pontos de $10^3$ a $10^{14}$ GeV.

> ⚠️ Estas tabelas ficam **acima** da parametrização de referência
> Gandhi–Quigg–Reno–Sarcevic (1998) por fatores de ~7 a ~90, dependendo da energia
> e do modelo. O notebook 01 §2.2 mostra a comparação. Isso ainda **não** foi
> explicado — é o exercício 6 do notebook 01, e uma pendência real do HADROS3.

## Para os alunos

Comece pelo notebook 01 e faça os exercícios antes de olhar o código de produção.
Depois abra [`hadros3/dis_sampler.py`](../hadros3/dis_sampler.py) e ache, no código
real, cada fórmula que você acabou de implementar — as referências
`arquivo.py:linha` nos comentários dos notebooks são o seu mapa. Quando as duas
coisas se encaixarem na sua cabeça, o notebook cumpriu a função dele.

Leitura de apoio: [`docs/Theory/HADROS3_Physics_Theory.tex`](../docs/Theory/HADROS3_Physics_Theory.tex).
