# HADROS3 — Estudo de Implementação de Event Generation

**Status:** estudo de arquitetura e plano de implementação  
**Data:** 2026-07-13  
**Escopo proposto:** H3-W10, geração do estado final com PYTHIA 8 entre POWHEG e GEANT4

## 1. Decisão executiva

A aba **Event Generation** deve deixar de ser um placeholder e passar a representar uma etapa física bem delimitada:

```text
POWHEG (processo duro em LHE)
    -> Event Generation (chuveiro + hadronização + decaimentos em PYTHIA 8)
    -> GEANT4 (transporte das partículas no meio)
```

A implementação recomendada é um backend C++17 baseado em **PYTHIA 8**, com saída primária em **HepMC3** e saídas normalizadas JSONL/JSON para a interface web e para o futuro adaptador GEANT4. O estágio só poderá executar sobre LHE real e validado produzido por `real_smoke` ou `real_free`; cartões de `dry_run` não são eventos e devem ser recusados.

GEANT4 não deve ser incorporado à aba Event Generation. O primeiro produz o estado final primário; o segundo transporta esse estado pelo material. Manter essa fronteira torna reproduzibilidade, pesos, falhas e validação física auditáveis.

O recorte físico inicial deve ser:

- DIS de corrente carregada já produzido pelo POWHEG local;
- referencial local da matéria no ponto de interação;
- importação LHE, chuveiros, hadronização e decaimentos configuráveis;
- um evento PYTHIA para cada evento LHE aceito, sem reamostragem implícita;
- preservação separada do peso do gerador e dos pesos astrofísicos/observacionais;
- saída pronta para consumo por GEANT4, mas sem transporte em Kerr ou no meio nesta etapa.

## 2. Estado atual auditado

O repositório já contém quase toda a cadeia anterior, mas nenhuma implementação de Event Generation.

| Componente | Estado observado | Consequência |
|---|---|---|
| `hadros3/config.py` | A aba possui apenas `mode=placeholder_disabled` e `planned_model=POWHEG_PYTHIA_future` | Não há configuração executável |
| `hadros_web.py` | O status de Event Generation é sempre `pending` | A interface não inspeciona produtos nem execução |
| Renderização web | A aba cai no renderer genérico | Não há botão, resultados, diagnósticos ou mensagens de disponibilidade |
| API/CLI | Há `/api/powheg` e opções CLI de POWHEG; não há equivalentes para Event Generation | Não existe orquestração do estágio |
| `hadros3/paths.py` | O layout termina em `POWHEG/` antes de `Dashboard/` | Falta diretório oficial e política de limpeza/invalidação |
| `hadros3/pipeline.py` | Aceita `powheg_summary`, mas não `event_generation_summary` | Produtos e proveniência não entram no resumo global |
| Proveniência | `pythia_invoked=false` e `expensive_event_generation_invoked=false` estão fixados em vários pontos | O estado real futuro não pode ser representado corretamente |
| POWHEG | `real_smoke` e `real_free` geram, validam e analisam LHE | Existe uma entrada concreta e versionável para o novo estágio |
| Ambiente atual | PYTHIA 8.312 e HepMC3 3.03.01 foram instalados no ambiente micromamba `dis`; geração C++ e escrita HepMC3 passaram em smoke local | As dependências de desenvolvimento estão disponíveis; bootstrap reproduzível e manifest de versões ainda são entregas de EG-1 |
| Testes | Há boa cobertura dos contratos de POWHEG e do parser LHE | Esses fixtures podem iniciar o desenvolvimento sem executar uma produção cara |

O contrato existente em `docs/PIPELINE_STAGE_CONTRACT.md` exige que cada etapa leia produtos oficiais anteriores, escreva apenas no próprio diretório, não altere etapas anteriores, registre proveniência e distinga claramente proxies de resultados físicos. Event Generation deve seguir o mesmo padrão.

Há uma divergência de numeração documental a corrigir: `PIPELINE_STAGE_CONTRACT.md` ainda chama Event Generation de H3-W9, enquanto a teoria vigente reserva H3-W9a/H3-W9b para POWHEG e chama PYTHIA de **H3-W10**, GEANT4 de H3-W11, Photon Transport de H3-W12 e Spectra de H3-W13. Este estudo adota a numeração da teoria vigente; EG-0 deve alinhar o contrato antigo.

## 3. Fronteira física do estágio

### 3.1 O que Event Generation fará

Para cada evento duro LHE válido:

1. importar o registro Les Houches, inclusive `SCALUP`, `XWGTUP`, cores, mães e estratégia `IDWTUP`;
2. aplicar a configuração controlada de chuveiro inicial/final compatível com a interface POWHEG;
3. hadronizar os partons e remanescentes de feixe;
4. executar os decaimentos habilitados;
5. serializar o registro completo e a lista de partículas finais estáveis;
6. validar conservação, integridade, pesos e rastreabilidade;
7. gerar sumários e diagnósticos para a aba web.

### 3.2 O que não fará

- não recalculará o espalhamento duro;
- não escolherá candidatos nem imagens gravitacionais;
- não multiplicará ou reinterpretará pesos do Observer Bridge;
- não propagará secundários na métrica de Kerr;
- não modelará perdas, interações ou deposição de energia no toro;
- não executará GEANT4, transporte de fótons ou síntese de espectro observado.

### 3.3 Referencial e unidades

O LHE do POWHEG descreve o espalhamento no **referencial local do alvo/matéria** usado para construir o cartão fixed-target. PYTHIA deve operar nesse mesmo referencial, em GeV e milímetros quando aplicável ao formato de vértices. O adaptador deve registrar explicitamente:

- `generator_frame = local_matter_tetrad`;
- a identificação e os quatro-vetores da tétrada local que originou o evento, quando disponíveis no request upstream;
- `momentum_unit = GeV` e `length_unit = mm`;
- a transformação local ↔ coordenadas de Kerr como metadado, sem aplicá-la silenciosamente ao evento.

Sem esse contrato, um evento fisicamente correto em PYTHIA pode ser posteriormente orientado ou transportado no referencial errado.

## 4. Entradas oficiais e validação de pré-condições

### 4.1 Entradas

O estágio deve consumir somente:

- `POWHEG/powheg_summary.json`;
- `POWHEG/powheg_event_requests.jsonl`;
- cada `POWHEG/powheg_lhe/<powheg_request_id>/pwgevents.lhe` declarado nos requests;
- snapshot validado da seção `event_generation` da configuração;
- metadados upstream referenciados pelo request, sem reler e reranquear candidatos.

### 4.2 Pré-condições bloqueantes

A execução real deve falhar cedo e com mensagem específica quando:

- `powheg_run_mode` não for `real_smoke` nem `real_free`;
- `powheg_lhe_generated` não for verdadeiro;
- um LHE estiver ausente, vazio ou sem fechamento válido;
- o número de eventos contado divergir do resumo POWHEG;
- o hash de um arquivo divergir do manifest capturado no início;
- PYTHIA 8 ou HepMC3 não estiverem compilados/disponíveis;
- uma combinação de opções não suportada for solicitada;
- o LHE não puder ser inicializado pelo backend, inclusive falhas de cor/remanescente.

Não deve existir fallback silencioso para evento LHE partônico, resultado sintético ou sucesso parcial. Execução parcial pode ser preservada para diagnóstico, mas o status global deve ser `failed` ou `partial_failed` e listar cada request afetado.

## 5. Arquitetura proposta

### 5.1 Camadas

```text
hadros-web / CLI / Makefile
        |
        v
hadros3/event_generation.py
  - valida configuração e manifest
  - cria jobs isolados
  - invoca backend
  - agrega HepMC/JSONL
  - calcula validações e plots
        |
        v
bin/hadros3_event_generator (C++17)
  - PYTHIA 8 LHA input
  - POWHEG matching/veto
  - shower, hadronização, decaimentos
  - HepMC3 + resumo canônico por evento
```

O Python deve permanecer responsável pela orquestração, produtos, plots e integração web. A chamada de PYTHIA e a conversão HepMC devem ficar no C++, evitando um segundo modelo do registro de eventos e reduzindo cópias de grandes amostras.

### 5.2 Novos arquivos principais

- `cpp/apps/hadros3_event_generator.cpp`: executável autocontido do estágio;
- `cpp/include/event_generation_contract.hpp`: estruturas e validações compartilháveis;
- `hadros3/event_generation.py`: manifest, execução, agregação e diagnósticos;
- `scripts/event_generation/bootstrap_pythia.py`: detecção/instalação reproduzível e manifest de versão;
- `tests/test_event_generation.py`: testes de contrato e numéricos;
- `tests/fixtures/event_generation/`: LHE mínimos válidos, incluindo pesos negativos e evento NLO;
- `docs/EVENT_GENERATION_BOOTSTRAP.md`: instalação e resolução de falhas.

O bootstrap deve fixar versões e hashes somente após validação das versões suportadas. O estudo não inventa um número de versão: a escolha será registrada no lock/manifest quando os pacotes forem efetivamente integrados.

### 5.3 Modos de execução

| Modo | Comportamento |
|---|---|
| `disabled` | Nenhuma execução; estado explícito e sem produtos falsos |
| `dry_run` | Valida dependências, configuração, manifest e LHE; não inicializa geração cara |
| `parton_check` | Importa o LHE com shower/hadronização/decaimentos desligados para validar o bridge |
| `real_smoke` | Primeiro request e no máximo dois eventos LHE, com toda a física configurada |
| `real_free` | Todos os requests/eventos permitidos pelos limites explícitos |

`parton_check` é deliberadamente um diagnóstico, nunca um produto hadronizado. Todos os seus resumos devem carregar `hadronization_invoked=false`.

### 5.4 Configuração proposta

```json
{
  "event_generation": {
    "mode": "disabled",
    "backend": "pythia8",
    "input_source": "powheg_lhe",
    "shower_mode": "powheg_vetoed",
    "isr_enabled": false,
    "fsr_enabled": true,
    "hadronization_enabled": true,
    "decays_enabled": true,
    "mpi_enabled": false,
    "random_seed": 48001,
    "seed_mode": "base_plus_request_and_lhe_event",
    "max_requests": 1,
    "max_events_per_request": 2,
    "write_hepmc3": true,
    "write_full_event_jsonl": false,
    "write_final_particles_jsonl": true,
    "failure_policy": "fail_stage"
  }
}
```

Os defaults de segurança devem ser `disabled`, um request e dois eventos. `real_free` pode remover os clamps apenas quando o usuário fornecer limites positivos. Tune, PDF do shower, QED radiation, decaimentos de hádrons de vida longa e MPI precisam aparecer como opções versionadas, não como defaults ocultos do PYTHIA.

Para DIS alimentado por POWHEG, `mpi_enabled=false` é a política inicial. A validação EG-3 encontrou que o ISR genérico do PYTHIA 8 no LHE fixed-target produz resíduo relativo de quatro-momento de `2.61e-5`; com ISR desligado e FSR/hadronização/decaimentos ativos, o resíduo medido foi `9.47e-10`. Assim, o recorte inicial aceito usa `isr_enabled=false` e FSR vetado por `SCALUP`. ISR permanece experimental até existir um hook DIS fixed-target que passe a mesma tolerância. O matching deve registrar essa política no resumo; simplesmente ligar shower sem registrar o veto não é aceitável para NLO+PS.

## 6. Identidade, pesos e normalização

### 6.1 Identidade bijetiva

Cada evento deve preservar:

```text
powheg_request_id
lhe_event_index
event_generation_event_id = <powheg_request_id>:<lhe_event_index>
interaction_id
source_sample_id
primary_branch_id
```

Não deve haver um novo sorteio de candidato. A relação normal é 1:1 entre evento LHE consumido e evento PYTHIA produzido. Se no futuro houver oversampling, ele exigirá um campo `replica_index` e uma regra explícita de divisão de peso.

### 6.2 Três famílias de peso que não podem ser misturadas

O contrato deve carregar, em campos separados:

1. **peso do gerador:** `XWGTUP`, `IDWTUP`, pesos alternativos LHE e declaração `XSECUP`;
2. **peso físico astrofísico:** amostragem da fonte, interação/óptica e demais fatores físicos upstream;
3. **peso de seleção/observação:** score e pesos do Observer Bridge/ramo de imagem.

PYTHIA deve propagar o peso do evento LHE sem convertê-lo em probabilidade de seleção. O score observacional nunca deve ser gravado em `GenEvent::weights()` como se fosse peso NLO. Um produto de análise poderá combiná-los posteriormente, com fórmula e normalização declaradas.

Para `IDWTUP=+4/-4`, o estimador por eventos continua sendo a média de `XWGTUP`, não a soma. Duplicar a amostra LHE deve deixar esse estimador invariante. Eventos de peso negativo devem ser preservados.

Os resultados de Event Generation serão **condicionais aos candidatos/imagens já selecionados**. Eles não representam por si só uma taxa all-sky não enviesada.

## 7. Produtos e contrato de dados

Adicionar `EVENT_GENERATION_DIR = "EventGeneration"` em `hadros3/paths.py`, com limpeza própria e migração de produtos conhecidos.

```text
EventGeneration/
  event_generation_manifest.json
  event_generation_jobs.jsonl
  event_generation_summary.json
  event_generation_summary.csv
  event_generation_validation_report.json
  event_generation_events.hepmc3
  event_generation_events_summary.jsonl
  event_generation_final_particles.jsonl
  event_generation_particle_content.json
  event_generation_multiplicity.png
  event_generation_energy_spectrum.png
  event_generation_species.png
  event_generation_conservation.png
  event_generation_event_view.html
  jobs/<powheg_request_id>/
    pythia.cmnd
    event_generation.log
    events.hepmc3
    events_summary.jsonl
    final_particles.jsonl
```

O manifest deve conter hashes SHA-256 dos LHE e da configuração, versões de PYTHIA/HepMC/compilador, commit HADROS3, seeds, opções efetivas e caminhos relativos. Não incluir timestamps na representação canônica usada para o teste de determinismo.

### 7.1 Registro mínimo por evento

- IDs completos e `lhe_event_index`;
- seed efetivo;
- `XWGTUP`, `IDWTUP`, `XSECUP` e pesos alternativos;
- `SCALUP` e escala efetiva do shower;
- contagens antes/depois de shower, hadronização e decaimentos;
- soma de quatro-momento inicial e final;
- resíduos de energia/momento, carga e números quânticos testados;
- flags efetivas `shower_invoked`, `hadronization_invoked`, `decays_invoked`;
- status e mensagens de erro do PYTHIA;
- offset/identificador do evento correspondente em HepMC3.

### 7.2 Registro mínimo por partícula final

- `event_generation_event_id`, índice e PDG;
- status canônico, mães e filhas quando preservadas;
- quatro-momento, massa e vértice;
- carga em unidades de `e/3` para teste inteiro;
- classificação estável/visível/neutrino/fóton/hádron;
- referencial e unidades.

## 8. Integração com hadros-web

A aba deve ganhar um renderer próprio, seguindo o padrão já usado por POWHEG:

- painel de disponibilidade do backend e versões detectadas;
- painel de entrada com número de LHE, requests e hashes válidos;
- controles dos campos da seção `event_generation`;
- botão contextual: “Validate Event Generation”, “Run Event Generation Smoke” ou “Run Event Generation”;
- barra de status e erro por request;
- tabela de contagens, pesos e conservação;
- histogramas de multiplicidade, espécies e energia;
- visualizador de um evento selecionável;
- links para HepMC3, JSONL, manifest, logs e validação.

Mudanças de orquestração:

- novo `POST /api/event-generation`;
- novas opções `--event-generation-dry-run`, `--event-generation-real-smoke` e `--event-generation-real-free`;
- novos targets `make event-generation-dry-run`, `make event-generation-real-smoke` e `make event-generation-real-free`;
- status da pipeline calculado pela existência e conteúdo de `event_generation_summary.json`, nunca fixado em `pending`;
- `render_hadros_web(..., event_generation_summary=...)` e `build_provenance(..., event_generation_summary=...)`;
- catálogo de outputs com todos os produtos do novo diretório.

O botão real deve ficar desabilitado, com explicação visível, enquanto faltar backend ou LHE real. `dry_run` continuará disponível para diagnosticar a instalação.

## 9. Invalidação e atomicidade

Cada estágio escreve somente no próprio diretório. Antes de executar, Event Generation cria uma área temporária no próprio output e só publica manifest/resumo final após sucesso das validações obrigatórias.

Regras de invalidação:

- rerodar POWHEG invalida e limpa `EventGeneration/`, `GEANT4/`, `PhotonTransport/` e `Spectra/`;
- rerodar Event Generation invalida apenas seus produtos e os estágios posteriores;
- falha não altera `POWHEG/`;
- salvar uma configuração não executa nem apaga resultados; a interface apenas marca produtos existentes como `stale` quando o hash divergir;
- o resumo deve distinguir `not_run`, `unavailable`, `ready`, `running`, `ok`, `partial_failed`, `failed` e `stale`.

## 10. Testes numéricos de aceitação

As tolerâncias abaixo são critérios iniciais de engenharia. Qualquer relaxamento deve ser justificado com uma amostra patológica reproduzível e registrado na teoria/proveniência.

### EG-T1 — Bridge LHE e cardinalidade

- número de eventos de saída igual ao número de blocos LHE selecionados;
- bijeção de `(powheg_request_id, lhe_event_index)` para `event_generation_event_id`;
- zero IDs ausentes ou duplicados;
- hashes de entrada antes/depois idênticos;
- evento truncado, NaN ou LHE de `dry_run` obrigatoriamente rejeitado.

### EG-T2 — Reprodução partônica

Em `parton_check`, com shower, hadronização e decaimentos desligados:

- PDG, status relevante, cores e quatro-vetores duros recuperáveis do registro importado;
- `SCALUP` e `XWGTUP` idênticos ao LHE em precisão de serialização;
- resíduo relativo de quatro-momento
  `max(|ΔE|,|Δpx|,|Δpy|,|Δpz|)/max(|Ein|,1 GeV) <= 5e-8`;
- a massa do alvo usada pelo PYTHIA deve ser alinhada ao `EBMUP(2)` do
  contrato fixed-target; o alvo deve permanecer em repouso com
  `|p_z|/max(E,1 GeV) <= 1e-9`;
- nenhuma saída pode ser rotulada como hadronizada.

### EG-T3 — Conservação após geração completa

Somando todas as partículas finais estáveis do evento completo, inclusive invisíveis:

- resíduo relativo de quatro-momento `<= 5e-8`;
- carga elétrica conservada exatamente em unidades inteiras de `e/3`;
- todos os valores finitos;
- para cada partícula, `E >= 0` e `|E²-p²-m²|/max(E²,1 GeV²) <= 1e-8`;
- eventos que o backend abortar não podem entrar na amostra `ok`.

Conservação de número bariônico e leptônico deve ser relatada e testada por topologia, levando em conta remanescente de feixe e decaimentos; não deve ser aplicada como um contador ingênuo de PDGs a estados intermediários.

### EG-T4 — Hadronização e conteúdo físico

Com hadronização habilitada:

- nenhuma partícula final estável pode ser quark ou glúon;
- uma amostra DIS não vazia deve produzir ao menos um hádron final;
- o relatório deve separar léptons carregados, neutrinos, fótons, mésons e bárions;
- partículas desconhecidas e códigos PDG inválidos devem ser zero;
- ligar/desligar decaimentos deve alterar apenas o estágio esperado e permanecer explicitamente rotulado.

Não impor uma “multiplicidade correta” por um único número. A aceitação estatística deve comparar histogramas congelados de uma amostra de referência versionada usando média, quantis e distância KS/chi-quadrado com tolerância documentada.

### EG-T5 — Pesos e normalização

- `XWGTUP` preservado evento a evento, inclusive sinal;
- `IDWTUP` e `XSECUP` preservados no resumo;
- `raw_weight_sum_is_cross_section=false`;
- para `IDWTUP=±4`, estimador igual à média dos pesos a `1e-12` relativo;
- duplicar integralmente o fixture LHE não altera esse estimador a `1e-12` relativo;
- pesos astrofísicos e score observacional não modificam `XWGTUP`;
- nenhum evento de peso negativo é descartado.

### EG-T6 — Determinismo

Com mesmo binário, configuração, LHE e seed:

- JSONL canônico e sumário físico byte a byte idênticos;
- mesmos PDGs, genealogia e quatro-vetores;
- HepMC3 comparado por conteúdo canônico, ignorando apenas metadados explicitamente não físicos;
- mudar a seed deve alterar pelo menos um evento showered em uma amostra de smoke, mantendo os invariantes.

### EG-T7 — Matching POWHEG + shower

- escala e política de veto registradas por evento;
- nenhum emission scale aceito acima do limite de matching definido, dentro da precisão numérica;
- teste A/B `parton_check` versus shower mostra que o processo duro e seu peso permanecem identificáveis;
- fixture Born e fixture NLO com emissão real devem ambos inicializar;
- warnings de matching acima de um limiar configurado tornam o smoke inválido.

### EG-T8 — Web, proveniência e isolamento

- API rejeita configuração inválida com erro 4xx e mensagem estruturada;
- aba exibe indisponibilidade sem simular sucesso;
- `pythia_invoked=true` somente após execução real;
- `hadronization_invoked` reflete a opção efetiva;
- `geant4_invoked=false`, `photon_transport_invoked=false` e `spectra_invoked=false` permanecem verdadeiros no estágio;
- hashes de todos os arquivos oficiais anteriores não mudam;
- status da pipeline muda de `pending/ready` para `done` apenas com resumo `ok` e validação aprovada.

### EG-T9 — Integração opt-in

Um teste caro, protegido por `HADROS3_RUN_REAL_EVENT_GENERATION_TEST=1`, deve executar:

```text
fixture/POWHEG real LHE -> PYTHIA smoke -> HepMC3 -> parser -> validation
```

Após estabilização, um segundo teste opt-in executará a cadeia pequena completa:

```text
Source -> Kerr geodesics -> DIS -> Observer Bridge -> Image Branch
       -> POWHEG NLO real smoke -> Event Generation real smoke
```

Aceitação: pelo menos um evento completo, zero falhas, todos os invariantes EG-T1 a EG-T8 e produtos/proveniência consistentes.

### EG-T10 — Orçamento operacional

O smoke de dois eventos deve ter limites explícitos de tempo e memória no CI opt-in. Para `real_free`, o resumo deve publicar eventos/s, pico de memória aproximado, bytes/evento HepMC3 e estimativa antes de iniciar. Não se deve definir ainda um limite rígido de produção sem medir o backend integrado.

## 11. Campanha de implementação, ponto a ponto

Cada item só recebe `COMPLLETO` depois de seus testes numéricos associados passarem. A ordem evita depurar física, formato e interface simultaneamente.

1. **EG-0 — Congelar o contrato de dados — COMPLLETO**  
   Definir schemas de job, evento, partícula, manifest, pesos e estados. Atualizar `PIPELINE_STAGE_CONTRACT.md`.  
   **Aceitação:** fixtures válidos/inválidos e testes de schema, IDs, hashes e normalização EG-T1/EG-T5.

2. **EG-1 — Bootstrap reproduzível de PYTHIA 8 e HepMC3 — COMPLLETO**  
   Detector de disponibilidade, versões fixadas, build C++ mínimo e modo `dry_run`.  
   **Aceitação:** build limpo, versão/hash no manifest, erro controlado sem dependência, nenhuma falsa invocação.

3. **EG-2 — Importador LHE em `parton_check` — COMPLLETO**  
   Ler todos os jobs reais, preservar metadados e escrever HepMC3/JSONL sem shower.  
   **Aceitação:** EG-T1, EG-T2, pesos positivos/negativos e isolamento upstream.

4. **EG-3 — Matching e chuveiros — COMPLLETO**  
   Implementar política de veto POWHEG, ISR/FSR e auditoria das escalas.  
   **Aceitação:** EG-T3, EG-T6 e EG-T7 em fixtures Born e NLO.

5. **EG-4 — Hadronização e decaimentos — COMPLLETO**  
   Habilitar remanescente, fragmentação e política explícita de decaimentos.  
   **Aceitação:** EG-T3, EG-T4, conservação por topologia e referência estatística versionada.

6. **EG-5 — Produtos, diagnósticos e visualizador — COMPLLETO**  
   Agregação, plots, tabelas e navegador HepMC/JSONL.  
   **Aceitação:** contagens cruzadas entre HepMC, JSONL e resumo; testes de arquivos e conteúdo.

7. **EG-6 — Integração hadros-web/CLI/Makefile — COMPLLETO**  
   Renderer próprio, endpoint, botões, status, catálogo e modos de execução.  
   **Aceitação:** testes de API/UI, modo indisponível, smoke e `real_free` com clamps corretos.

8. **EG-7 — Pipeline, invalidação e proveniência — COMPLLETO**  
   Encadear summaries, diretórios e limpeza downstream; remover flags fixas onde o novo estágio é ativo.  
   **Aceitação:** EG-T8 e hashes upstream invariantes.

9. **EG-8 — Validação integrada e teoria — COMPLLETO**  
   Executar EG-T9, documentar equações, pesos, matching, referencial e limitações em `HADROS3_Physics_Theory.tex`; regenerar PDF.  
   **Aceitação:** testes unitários completos, ambos os smokes opt-in, `make theory`, `make check` e `make validate`.

## 12. Riscos e decisões que precisam de evidência numérica

| Risco | Impacto | Controle proposto |
|---|---|---|
| Tratamento de remanescente em LHE DIS fixed-target | Falha de cor ou estado final não físico | Fase EG-2 com fixtures Born/NLO e auditoria do event record |
| Matching POWHEG–PYTHIA incorreto | Double counting ou emissões acima da escala | Veto explícito, histograma de escalas e EG-T7 |
| UHE e grande razão de escalas | Instabilidade numérica | Valores finitos, on-shell, conservação `5e-8` e amostras por década de energia |
| Pesos NLO negativos | Viés se descartados ou usados como probabilidade | Preservar sinal, testes de duplicação e normalização |
| Mistura entre peso do gerador e score do observador | Taxas fisicamente erradas | Namespaces separados e nenhuma multiplicação nesta etapa |
| Orientação/referencial local | GEANT4 receber eixos errados | Tétrada e transformação como metadados obrigatórios |
| Arquivos HepMC muito grandes | Saturação de disco/interface | Limites, estimativa prévia, JSONL final opcional e streaming |
| Defaults ocultos do PYTHIA | Resultados não reproduzíveis | Dump completo de settings efetivos e versões no manifest |
| Sucesso parcial interpretado como completo | Produtos inconsistentes | Publicação atômica e estados `partial_failed/failed` |

## 13. Critério de conclusão do estágio

Event Generation estará implementado — e poderá substituir `placeholder_disabled` — somente quando:

- houver backend PYTHIA 8 + HepMC3 reproduzível;
- LHE real de todos os requests selecionados for consumido sem alterar upstream;
- os contratos de IDs, pesos, referencial e unidades estiverem materializados nos produtos;
- conservação e integridade passarem nas tolerâncias definidas;
- matching, shower, hadronização e decaimentos estiverem explicitamente registrados;
- smoke e cadeia opt-in passarem numericamente;
- a aba web executar, diagnosticar e exibir produtos reais;
- pipeline, invalidação, catálogo e proveniência refletirem o estado real;
- teoria e limitações forem atualizadas;
- os itens EG-0 a EG-8 estiverem marcados `COMPLLETO` com evidência dos testes.

## 14. Evidência final de aceitação

A campanha EG-0--EG-8 foi concluída em 2026-07-13. O contrato versionado está em `schemas/event_generation_contract.schema.json`; o backend detectado foi PYTHIA 8.312 com HepMC3 3.03.01. A suíte normal passou por `make validate` (94 testes coletados), os smokes reais LO e NLO passaram com determinismo byte a byte, variação de seed e política de decaimentos (12 testes de Event Generation) e a cadeia NLO opt-in completa, de UHE Source até HepMC3, passou em diretório temporário. Em ambos os smokes aceitos houve hádrons finais, nenhum parton final, matching abaixo de `SCALUP`, conservação de quatro-momento abaixo de `5e-8` e residual on-shell abaixo de `1e-8`.

O limite físico remanescente não é ocultado: ISR genérico fixed-target continua experimental e desligado, pois o teste mediu residual `2.61e-5`; a configuração aceita usa FSR com veto POWHEG, hadronização e decaimentos. O estágio independente H3-W11 GEANT4 agora consome este HepMC3 sem alterar H3-W10 e foi validado no domínio suportado. O evento UHE corrente excede o teto físico documentado e é recusado antes de `BeamOn`; isso permanece um bloqueio científico explícito, não uma falha do H3-W10.

Em 2026-07-14 foi acrescentada uma regressão em `10^4 GeV`. Ela revelou que
o `EBMUP(2)=0.938272 GeV` do POWHEG e a massa default arredondada
`0.938270 GeV` do PYTHIA davam ao alvo nominalmente parado um momento espúrio
de `1.937 MeV/c`. O backend agora alinha explicitamente a massa do próton ao
contrato LHE e valida o repouso do alvo. No evento real que motivou a correção,
o residual de quatro-momento caiu de `2.0076e-7` para `1.9286e-11` e o residual
do referencial do alvo ficou em `1.9916e-11`, ambos aprovados sem relaxar as
tolerâncias.
