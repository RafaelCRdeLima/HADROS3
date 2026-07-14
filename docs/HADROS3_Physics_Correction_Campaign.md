# HADROS3 Physics Correction Campaign

Data de abertura: 2026-07-13  
Base auditada: `8395440`  
Documento de referência: `docs/Theory/HADROS3_Physics_Theory.pdf` (Theory 1.2)

## Regra de fechamento

Um ponto só recebe a marca `COMPLLETO` (grafia solicitada) quando:

1. a implementação e o documento teórico descrevem o mesmo modelo;
2. há teste numérico específico para a correção;
3. o teste específico e a suíte de regressão passam;
4. a evidência numérica e os arquivos alterados são registrados neste documento.

`COMPLLETO` não significa que uma aproximação virou física exata. Quando uma aproximação é mantida, ela deve ser explicitamente nomeada, limitada e coberta por teste.

## Pontos da campanha

### COMPLLETO — P01 — Profundidade óptica covariante no referencial da matéria

- Problema: o DIS usa uma distância espacial em uma fatia `t=constante` de Boyer–Lindquist. Para um raio nulo em matéria com quadrivelocidade `u`, o incremento físico deve usar `dell_com = -(u.k) dlambda` e `dtau = n_com sigma(E_com) dell_com`.
- Correção: registrar o intervalo afim e o quadrimomento por segmento; calcular energia e comprimento no mesmo referencial da matéria; manter qualquer modo legado apenas como opção explicitamente diagnóstica.
- Aceite numérico:
  - limite de Minkowski estático reproduz `dtau = n sigma dl` com erro relativo `<= 1e-10`;
  - invariância sob reescala `k -> c k`, `dlambda -> dlambda/c` com erro relativo `<= 1e-10`;
  - backend C++ e referência Python concordam em `tau` com erro relativo `<= 5e-10` em caso determinístico;
  - todos os incrementos de tau são finitos e não negativos.

Evidência de fechamento (2026-07-13):

- `KerrGeodesic::step_adaptive` agora retorna o intervalo afim efetivamente aceito pelo RKF adaptativo.
- Os segmentos registram `affine_parameter_step_rg`, `momentum_affine_normalization_gev` e o comprimento comóvel ZAMO de auditoria.
- Os backends DIS Python e C++ calculam `Delta tau` com `(-u.k/E_norm) Delta lambda r_g`; `dl_segment_rg` permanece apenas diagnóstico.
- Segmentos antigos sem metadados afins falham com mensagem para regenerar ForwardGeodesics, em vez de degradar silenciosamente para a aproximação espacial.
- Teste plano: erro relativo `< 1e-12` para `E_local=E_norm` e comprimento esperado de `3.25 r_g`.
- Teste de reescala afim por fator 37: erro relativo `< 1e-12`.
- Comparação integrada Python/C++ passou com tolerância determinística `5e-12` para `tau`, densidade, sigma e `Delta tau` máximo.
- `make cpp`: aprovado sem warnings.
- Testes específicos: `14 passed in 195.68s`.
- Suíte completa: `69 passed, 1 warning in 258.18s`; o warning é apenas uma legenda vazia de diagnóstico do Observer Bridge.
- `HADROS3_Physics_Theory.pdf` foi regenerado com a integral covariante e a aproximação de quadratura de ponto médio.

### COMPLLETO — P02 — Execução POWHEG declarada NLO, mas configurada como Born/LO

- Problema: os cards contêm `LOevents 1`, que zera contribuições reais e virtuais, enquanto o PDF declara NLO.
- Correção: criar modos físicos inequívocos (`lo_smoke` e `nlo`) e impedir que um produto LO seja rotulado NLO. O modo NLO deve executar sem `LOevents 1` e validar conteúdo/metadata NLO.
- Aceite numérico:
  - card LO contém a opção Born-only e provenance `perturbative_order=LO`;
  - card NLO não contém Born-only e provenance `perturbative_order=NLO`;
  - smoke LO/NLO conserva quadrimomento por evento com resíduo relativo `<= 5e-8`; este limite de 50 ppb cobre o condicionamento do boost laboratório UHE do backend Fortran, mas rejeita a amplificação do reshuffling finito;
  - smoke NLO real passa somente com integração real/virtual concluída e ao menos um LHE válido; ausência do backend deve falhar claramente, nunca degradar silenciosamente para LO.

Evidência de fechamento (2026-07-13):

- Nova configuração explícita `perturbative_order=LO|NLO`; o padrão é `LO` e é rotulado Born-only em card, resumo, requests, validação e provenance.
- O modo NLO omite `LOevents 1`; uma checagem cruzada aborta se card e metadata divergirem.
- O card agora ativa `fixed_target 1` e `masslesslhe 1`.
- O adaptador POWHEG DIS não executa o reshuffler de massas finitas quando o LHE foi solicitado massless. No caso UHE testado, isso reduziu o resíduo relativo de quadrimomento de `7.57e-5` para `1.70e-8`.
- Smoke NLO real: `1 passed in 2.93s`; o log confirma `LOevents absent`, quatro subprocessos Born, seis subprocessos reais CC e integração de btilde/real/virtual. O evento aceito contém emissão real de glúon e três partículas finais.
- Suíte POWHEG normal: `11 passed, 1 skipped in 6.93s`; o único skip é justamente o teste NLO real opt-in, executado separadamente e aprovado.
- PDF regenerado: o padrão LO e o NLO opcional agora são descritos separadamente, sem rotular o smoke LO como NLO.

### COMPLLETO — P03 — Semântica e normalização dos pesos LHE

- Problema: o parser ignora `<init>`, `IDWTUP` e `XSECUP`; o PDF afirma incorretamente que a seção de choque é a soma bruta de `XWGTUP`.
- Correção: analisar o bloco `<init>`, registrar a estratégia de pesos e calcular estimadores conforme `IDWTUP`.
- Aceite numérico:
  - parser recupera `IDWTUP`, `XSECUP`, `XERRUP`, `XMAXUP` e `LPRUP` de fixture conhecida exatamente;
  - duplicar N vezes uma amostra de eventos não multiplica a seção de choque estimada;
  - pesos positivos e negativos produzem média e incerteza esperadas em fixture analítica com erro `<= 1e-12`.

Evidência de fechamento (2026-07-13):

- O parser lê o header `<init>` e todos os campos `IDBMUP`, `EBMUP`, `PDFGUP`, `PDFSUP`, `IDWTUP`, `NPRUP`, `XSECUP`, `XERRUP`, `XMAXUP` e `LPRUP`.
- `XSECUP` é somado por processo; `XERRUP` é combinado em quadratura.
- Para `IDWTUP=+/-4`, o estimador por eventos é a média assinada de `XWGTUP`; a soma bruta é marcada `raw_weight_sum_is_cross_section=false`.
- Fixture com pesos `[3,-1,2,0]`: média `1.0`, erro padrão `sqrt(10/12)` e invariância exata ao duplicar a amostra sete vezes.
- Testes POWHEG após a correção: `12 passed, 1 skipped in 6.85s`; o skip NLO real foi executado separadamente no P02.
- O capítulo de pesos do Theory foi atualizado com a normalização `IDWTUP=-4`.

### COMPLLETO — P04 — Proveniência científica 1.1 versus Theory/Physics 1.2

- Problema: `provenance.py` contradiz `VERSION.json` e o PDF quanto a versões, commit, data e estágios implementados.
- Correção: estabelecer uma fonte única de versão e validar automaticamente a coerência PDF/TeX/config/provenance.
- Aceite numérico:
  - teste compara todos os campos de versão e falha diante de qualquer divergência;
  - H3-W8b aparece entre os estágios implementados;
  - commit compatível é ancestral do HEAD.

Evidência de fechamento (2026-07-13):

- `provenance.py`, `VERSION.json` e os macros do Theory agora concordam em Software 0.9.0, Physics/Theory 1.2, pipeline H3-W9b, data 2026-06-28 e commit compatível `924b632`.
- H3-W8b foi incluído explicitamente nos estágios implementados.
- O teste estático compara todos esses campos e executa `git merge-base --is-ancestor 924b632 HEAD`.
- Testes de release/provenance: `7 passed in 0.22s`.

### COMPLLETO — P05 — Fórmula de `Qmax` inconsistente

- Problema: PDF/bootstrap usam `2 sqrt(m_N E)`; o driver usa `sqrt(2 m_N E)=sqrt(s)`.
- Correção: adotar uma única definição cinemática e gerar bootstrap e cards pela mesma função/fonte.
- Aceite numérico:
  - para `E=1e9 GeV`, `Qmax=min(sqrt(2 m_N E),1e5 GeV)` com erro relativo `<= 1e-12`;
  - `Qmax <= sqrt(s)` em varredura logarítmica de `1e3` a `1e14 GeV`;
  - bootstrap e driver concordam bit a bit no valor formatado.

Evidência de fechamento (2026-07-13):

- A fonte Python compartilhada `powheg_kinematics.py` define `Qmax=min(sqrt(2 m_N E), 1e5 GeV)` e é consumida diretamente pelo bootstrap.
- Para `E=1e9 GeV`, o teste reproduz `sqrt(2*0.938272*1e9)` com tolerância relativa `1e-12`.
- Uma varredura logarítmica de `1e3` a `1e14 GeV` confirma `Qmax <= sqrt(s)` em todos os pontos.
- O card produzido pelo driver C++ e o card do bootstrap geram exatamente a mesma string Fortran `Qmax` para `1e9 GeV`.
- Teste específico: aprovado; `make cpp`: aprovado sem warnings.

### COMPLLETO — P06 — Interpretação incorreta de `vtype=2`

- Problema: o PDF atribui `vtype` à seleção de sabor CC, mas no `nudis` ele controla contribuições de corrente neutra.
- Correção: remover a alegação; documentar `channel_type=3` como seletor CC e tratar `vtype` como irrelevante/não aplicável ao modo CC, conforme o backend.
- Aceite numérico:
  - teste de card confirma `channel_type=3`;
  - metadata não usa `vtype` para inferir sabor CC;
  - busca automatizada não encontra a alegação antiga no TeX gerado.

Evidência de fechamento (2026-07-13):

- Cards, requests e resumo registram `cc_process_selector=channel_type=3` e `incoming_lepton_pdg_id=12`.
- `vtype_physics_role=neutral_current_gamma_Z_content_not_CC_flavor` impede que o valor obrigatório do card seja interpretado como seletor de sabor CC.
- O relatório de conteúdo de partículas explica que `ih1=12`, e não `vtype`, seleciona o neutrino eletrônico incidente.
- O teste inspeciona card, metadata e Theory e confirma a ausência da alegação antiga.
- Aceitação conjunta P05/P06/NLO: `3 passed in 1.15s`.

### COMPLLETO — P07 — Viés dependente da ordem imposto por `max_interactions`

- Problema: após atingir o limite, caminhos posteriores são forçados a rejeição.
- Correção: realizar Bernoulli para todos os caminhos e, se necessário, subamostrar os aceitos por método uniforme e reprodutível.
- Aceite numérico:
  - permutações da mesma população produzem probabilidades marginais compatíveis;
  - em teste Monte Carlo, frequência de seleção por posição difere da média em menos de `5 sigma`;
  - mesma seed e mesma entrada preservam reprodutibilidade exata.

Evidência de fechamento (2026-07-13):

- Python e C++ executam o Bernoulli de todos os caminhos antes de aplicar o limite.
- Sucessos excedentes recebem prioridades pseudoaleatórias determinísticas derivadas de `seed`, `event_id` e stream; os menores valores formam uma amostra uniforme sem reposição.
- A metadata registra `max_interactions_cap_model=all_bernoulli_then_uniform_hash_priority_subsample` e `max_interactions_order_dependent=false`.
- Uma permutação não trivial dos 20 identificadores preservou exatamente o conjunto selecionado; mesma seed reproduziu exatamente os flags.
- Monte Carlo com 4.000 seeds, 20 caminhos e limite 5: todas as frequências posicionais ficaram a menos de `5 sigma` da expectativa de 1.000 seleções.
- Aceitação integrada Python/C++ e reprodutibilidade: `3 passed in 41.73s`; `make cpp`: aprovado sem warnings.

### COMPLLETO — P08 — Posição de interação dentro do segmento não segue `d tau`

- Problema: após selecionar o segmento, o ponto é aproximadamente uniforme, condicionado apenas a densidade positiva.
- Correção: amostrar a CDF intrassegmento de `n_com sigma(E_com) dell_com`, com quadratura/subdivisão controlada.
- Aceite numérico:
  - densidade constante produz distribuição uniforme, KS `p > 0.01` com seed fixa e amostra definida;
  - perfil linear/analítico reproduz média teórica com erro estatístico inferior a `5 sigma`;
  - nenhum ponto aceito fica fora do suporte do meio.

Evidência de fechamento (2026-07-13):

- Python e C++ constroem uma CDF intrassegmento com 128 células trapezoidais e invertem analiticamente/numericamente a densidade linear dentro da célula sorteada.
- Dentro da aproximação de ponto médio do segmento, `E_local`, `sigma` e a conversão afim são constantes; a densidade do meio é avaliada nos 129 nós, portanto a CDF é proporcional a `d tau/du` implementado.
- Perfil constante, 5.000 amostras: o teste KS fixo satisfaz `p > 0.01`.
- Perfil analítico `f(u)=2u`, 5.000 amostras: a média amostral concorda com `2/3` dentro de `5 sigma`, usando `Var(u)=1/18`.
- Testes integrados confirmam que nenhum ponto aceito fica fora do meio nos backends Python e C++.
- Aceitação P08: `3 passed in 33.47s`; `make cpp`: aprovado sem warnings.

### COMPLLETO — P09 — Proxies do Observer Bridge apresentados como pesos físicos

- Problema: visibilidade, LOS e redshift permanecem unitários mesmo quando habilitados.
- Correção: ou implementar os cálculos físicos correspondentes, ou tornar os nomes/estados inequivocamente diagnósticos e impedir configuração que sugira cálculo inexistente.
- Aceite numérico:
  - modo proxy registra `physics_model=false` e não altera peso ao alternar flags;
  - qualquer modo físico futuro deve passar casos Schwarzschild/Kerr de referência;
  - score final reproduz o produto documentado com erro relativo `<= 1e-12`.

Evidência de fechamento (2026-07-13):

- Visibility, redshift e line of sight foram renomeados para `unity_diagnostic_not_physics` no schema, preset, backend, resumo e provenance.
- Cada candidato e resumo registra `visibility_physics_model=false`, `redshift_physics_model=false`, `line_of_sight_physics_model=false` e que os flags legados não alteram os fatores unitários.
- Um teste executa o backend duas vezes, invertendo simultaneamente `redshift_weight_enabled` e `line_of_sight_check_enabled`; os vetores de `observer_weight` e `final_observation_score` permanecem exatamente iguais.
- Para todos os candidatos, o produto dos seis fatores reproduz `observer_weight`, e `physics_weight*observer_weight` reproduz o score, com tolerância relativa `1e-12`.
- Aceitação P09: `2 passed in 3.25s`; `make cpp`: aprovado sem warnings.

### COMPLLETO — P10 — Clustering de imagens múltiplas dependente da ordem

- Problema: a atribuição gulosa por centroide não representa componentes conectados e muda com a ordem dos raios.
- Correção: usar componentes conexos/DBSCAN com regra e métrica documentadas.
- Aceite numérico:
  - todas as permutações de fixtures pequenas geram os mesmos clusters;
  - cadeias transitivas `A-B-C` pertencem ao mesmo cluster quando `A-B` e `B-C` estão ligados;
  - separação acima do raio gera branches distintas.

Evidência de fechamento (2026-07-13):

- A atribuição por distância ao centroide foi substituída pelo grafo não dirigido de vizinhança em pixels e suas componentes conexas.
- Raios, componentes, melhor raio e branches recebem ordenação determinística com desempates explícitos.
- A metadata registra `image_clustering_model=pixel_radius_connected_components` e `image_clustering_order_dependent=false`.
- Todas as 24 permutações de uma fixture de quatro raios produziram exatamente `((20,), (10,11,12))`.
- A cadeia A–B–C foi mantida numa única branch embora A–C exceda o raio; um quarto raio separado permaneceu numa branch distinta.
- Aceitação P10 e regressão integrada: `2 passed in 3.26s`.

### COMPLLETO — P11 — Branch unitária sintética confundida com reconstrução multi-imagem

- Problema: na ausência da auditoria multi-imagem, o pixel mais próximo é convertido em branch de um raio.
- Correção: identificar o resultado como fallback não auditado e impedir alegações de multiplicidade.
- Aceite numérico:
  - fallback registra `multiplicity_audited=false` e `branch_is_synthetic_fallback=true`;
  - zero clusters explicitamente auditados nunca vira branch sintética;
  - seleção downstream preserva a distinção na provenance.

Evidência de fechamento (2026-07-13):

- Branches derivadas apenas do pixel mais próximo registram `multiplicity_audited=false` e `branch_is_synthetic_fallback=true` tanto no catálogo de branches quanto no registro primário downstream.
- Uma auditoria presente com `image_clusters=[]` preserva exatamente zero branches, registra `multiplicity_audited=true` e não recebe `primary_branch_id` sintético.
- Resumo e provenance registram separadamente `n_candidates_multiplicity_audited`, `n_synthetic_fallback_branches` e `multiplicity_claims_include_synthetic_fallback=false`.
- Fixture combinada: candidato sem auditoria produziu uma branch sintética; candidato explicitamente auditado com zero clusters permaneceu com zero.
- Aceitação P11 e regressão da seleção primária: `2 passed in 1.97s`.

### COMPLLETO — P12 — Descrição desatualizada do overlay da câmera

- Problema: o PDF descreve overlay pinhole não ray-traced, mas o padrão atual usa correspondência de pixels por geodésicas de Kerr.
- Correção: separar no documento o proxy geométrico de FOV, o background diagnóstico e o overlay por Kerr pixel match.
- Aceite numérico:
  - provenance e PDF concordam em `kerr_geodesic_pixel_match`;
  - teste de orientação confirma norte para cima em todas as figuras aplicáveis;
  - comparação de background confirma ausência de inversão vertical acidental.

Evidência de fechamento (2026-07-13):

- O Theory separa agora o FOV/pinhole diagnóstico, o background da câmera e o overlay padrão `kerr_geodesic_pixel_match`.
- Todos os produtos espaciais do Observer Overview aplicam uma única transformação `flip_y_for_north_up` tanto ao background quanto às coordenadas sobrepostas.
- O teste de pixel confirma exatamente `y_display = height - 1 - y_raw`, incluindo limites da imagem; o teste de base confirma topo `-e_theta/+z/NORTH` e produto escalar positivo com o screen-up esperado.
- A auditoria do background confirma SHA-256 idêntico à Camera Preview antes da transformação, `background_hash_match=true`, e registra explicitamente a transformação vertical aplicada.
- Os produtos de Observer Image Branches auditados registram norte no topo, sul embaixo e `display_y=image_height-source_y`.
- Aceitação P12: `5 passed in 4.98s`.

## Validação integrada final — COMPLETO

- Regenerar `HADROS3_Physics_Theory.pdf`.
- Executar toda a suíte de testes.
- Executar pelo menos um pipeline determinístico pequeno do source ao LHE.
- Registrar versões, seeds, tolerâncias, métricas e limitações remanescentes.
- Confirmar que nenhum item permanece com alegação física mais forte que a implementação.

Evidência de fechamento (2026-07-13):

- `make cpp`: todos os quatro backends C++ recompilados com `-Wall -Wextra -pedantic`, sem warnings.
- `make theory`: `HADROS3_Physics_Theory.pdf` regenerado em 39 páginas após três passagens LaTeX; referências resolvidas.
- Suíte final: `80 passed, 2 skipped in 154.92s`. Os dois skips são testes reais opt-in: NLO POWHEG isolado e pipeline source-to-LHE; ambos foram executados separadamente e aprovados.
- Warning diagnóstico remanescente da suíte foi eliminado; teste do painel de orientação: `1 passed in 3.40s`, sem warnings.
- Pipeline determinístico opt-in, `1 passed in 14.17s`:
  - fonte: 8 amostras, seeds de posição/direção `1122/3344`;
  - geodésicas: 5 caminhos e 160 segmentos, Kerr `a=0.9`;
  - DIS C++: seed `24680`, 1 interação aceita, nenhum ponto fora do meio;
  - Observer Bridge/Branches: 1 candidato e 1 branch auditada;
  - POWHEG: seed base `34100`, `perturbative_order=NLO`, 1 evento LHE, 3 partículas finais;
  - resíduo relativo máximo de quadrimomento: `2.3996896691173805e-10`, contra tolerância `5e-8`;
  - `<init>` válido com `IDWTUP=-4`, `XSECUP=9448.76 pb`; o único evento tem peso `12120 pb`, mantido como estimador de amostra unitária e não confundido com soma de seção de choque.
- O patch adaptador `masslesslhe` é aplicado de forma idempotente na árvore temporária de build POWHEG e registrado em `powheg_build_summary.json`; não depende da cópia externa ignorada pelo Git.
- `git diff --check`: aprovado.

Limitações explicitamente mantidas:

- quadratura de ponto médio para energia/momento dentro de cada segmento geodésico;
- velocidade do meio ZAMO como fallback, não solução GRMHD;
- visibility/redshift/LOS do Observer Bridge são diagnósticos unitários, não modelos físicos;
- score de branch é proxy de ranking, não magnificação de cáustica;
- LHE é hard-process partônico; PYTHIA, GEANT4, transporte de fótons e resposta de detector não foram executados.

## Registro de evidências

As evidências serão adicionadas abaixo de cada ponto no momento em que ele receber `COMPLETO`.
