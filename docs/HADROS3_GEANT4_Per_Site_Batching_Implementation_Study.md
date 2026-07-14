# HADROS3 — estudo de implementação do batching GEANT4 por sítio local

**Data:** 2026-07-14  
**Escopo:** correção do H3-W11 para transportar eventos H3-W10 situados em pontos distintos do toro, cada um com material e densidade local próprios.  
**Estado deste documento:** baseline por sítio implementado e numericamente aceito em 2026-07-14; refinamentos científicos de longo prazo permanecem identificados abaixo.

## 0. Relatório de implementação — COMPLLETO para o baseline por sítio

Foi implementada a solução operacional escolhida neste estudo:

- `hadros3/geant4_site_batch.py` planeja um job por identidade física `(interaction_id, powheg_request_id)`;
- cada job lê seu HepMC3 individual, cria seu próprio processo Geant4 e aplica exatamente a densidade do vértice DIS correspondente;
- `site_workers` limita o paralelismo externo; o backend continua serial (`threads = 1`) em cada sítio;
- a seed de cada job é um hash estável da seed base e da identidade/estado material do sítio, portanto independe da ordem dos workers;
- eventos, partículas escapadas e steps recebem `site_job_id`, `site_event_index` e `site_density_g_cm3` e são agregados em ordem global determinística;
- a publicação continua atômica: uma exceção em qualquer future remove o staging e preserva o produto H3-W11 anterior;
- o manifest superior registra todos os jobs, densidades, seeds, hashes de entrada e o modelo `per_site_subprocess`;
- os viewers macro e local usam a densidade do sítio selecionado, e não uma densidade global;
- `_merge_hepmc` agora mantém um único `GenRunInfo`, remove separadores vazios incompatíveis com o leitor usado e renumera os eventos globalmente.

### Aceitação numérica executada

Caso real `HADROS3_hadros_web_preview`, dois eventos em dois sítios:

| sítio | interação | densidade [g/cm³] | eventos | total de steps |
|---|---|---:|---:|---:|
| `H3G4SITE-000001` | `H3DIS-000020` | `4.0150632695502954e9` | 1 | 3,206,025 |
| `H3G4SITE-000002` | `H3DIS-000035` | `1.3576463675557237e9` | 1 | 2,923,919 |

O relatório conjunto aceitou:

```text
site_isolation_pass      = true
density_mapping_pass     = true
cardinality_pass         = true
energy_ledger_pass       = true
upstream_hash_unchanged  = true
```

A mesma amostra foi executada com um e dois workers. Os três produtos científicos agregados tiveram SHA-256 idêntico:

```text
geant4_events_summary.jsonl       aeea421edf7775eb9e6582400cd21763bf810de0504840b91053bb4555fabd69
geant4_escaped_particles.jsonl    404fb770fab23d8899ace21a3e002314d0d7882536b2ec201437bf623b5663f9
geant4_steps.jsonl                87385da090f21f1560f6fd0ee8bd069bf3b6a53a9a69b477756fd0ba7955c498
```

No ensaio de dois sítios, o tempo caiu de `5.6183 s` para `3.3297 s`, speedup `1.6873x`. O benchmark anterior de quatro processos permanece como caracterização de escalabilidade, não como garantia universal.

Testes focais executados após a implementação:

```text
34 passed, 2 skipped
```

Os skips são testes condicionais de artefatos opcionais, não falhas. Foram adicionados testes explícitos para duas densidades distintas, seeds distintas, isolamento dos jobs, preservação dos hashes e validade estrutural do HepMC3 agregado.

### Limite desta declaração

`COMPLLETO` aqui significa que o baseline corretivo solicitado — um cálculo Geant4 independente por sítio, com agregação determinística — está implementado. Estudos de convergência de tamanho do patch/cuts e modelos de matéria H/He ultradensa continuam sendo requisitos para elevar o resultado de baseline computacional a uma validação astrofísica de maior fidelidade; eles não invalidam o isolamento de densidades implementado.

## 1. Resumo executivo

O erro

```text
selected H3-W10 events have different local densities;
H3-W11 v1 requires one homogeneous-material job per density
```

é causado pela tentativa do orquestrador atual de colocar eventos pertencentes a sítios distintos dentro de um único `G4Run`, com um único `G4LogicalVolume` e um único `G4Material`. O guard é correto ao impedir que a densidade do primeiro sítio seja aplicada silenciosamente aos demais, mas o orquestrador está incompleto.

A correção recomendada é:

1. definir um **sítio de transporte** por `interaction_id`;
2. recuperar, para cada sítio, posição global, tétrada local, densidade, composição e tamanho do patch;
3. usar diretamente o HepMC3 individual de `EventGeneration/jobs/<powheg_request_id>/events.hepmc3`;
4. executar um backend GEANT4 serial e isolado por sítio;
5. agrupar no mesmo job somente réplicas do mesmo sítio e das mesmas condições materiais;
6. paralelizar os jobs independentes externamente, com número limitado de processos;
7. agregar os produtos em ordem determinística e publicar atomicamente a etapa H3-W11.

Esta estratégia é escolhida como baseline porque é fisicamente correta, simples de auditar, tolerante a falhas e já mostrou speedup real de `3.61x` com quatro jobs concorrentes. A otimização futura de representar vários patches separados dentro de uma única geometria Geant4 deve ser considerada somente depois que o baseline por subprocesso estiver validado.

## 2. Estado auditado da cadeia atual

Na amostra `output/HADROS3_hadros_web_preview`:

- o DIS aceitou 36 interações;
- o POWHEG executou 36 jobs, cada um com um evento LHE;
- o Event Generation produziu 36 eventos PYTHIA/HepMC3;
- cada evento conserva `powheg_request_id` e `interaction_id`;
- os sítios têm densidades diferentes, como devem ter.

Exemplos dos primeiros sítios selecionados:

| request | interaction | `r/r_g` | `theta` [rad] | `rho` [g/cm³] |
|---|---|---:|---:|---:|
| H3PWHG-000001 | H3DIS-000020 | 12.129880 | 1.354098 | `4.0150632696e9` |
| H3PWHG-000002 | H3DIS-000035 | 12.891905 | 1.245632 | `1.3576463676e9` |
| H3PWHG-000003 | H3DIS-000022 | 9.396934 | 1.592270 | `9.8124396815e9` |
| H3PWHG-000004 | H3DIS-000007 | 7.067599 | 1.680498 | `6.2734867889e9` |

### 2.1 Limitação exata do H3-W11 v1

`hadros3/geant4_transport.py::_resolve_density`:

1. lê os eventos H3-W10 até `geant4.max_events`;
2. procura cada `interaction_id` em `DIS/dis_accepted_interactions.jsonl`;
3. coleta `interaction_rho_g_cm3`;
4. exige que todas as densidades sejam iguais com tolerância relativa `1e-12`;
5. chama uma única vez `hadros3_geant4_transport`.

O backend C++ constrói uma única caixa `Patch` e, para `HADROS3_H_HE`, um único material com uma única densidade. Em seguida chama `BeamOn(N)`. Isso é coerente somente quando todos os `N` eventos representam réplicas do mesmo sítio material.

### 2.2 Defeito adicional descoberto no HepMC3 agregado

O arquivo global atual

```text
EventGeneration/event_generation_events.hepmc3
```

é formado por concatenação textual dos corpos dos arquivos individuais. Cada arquivo individual contém a declaração de metadados de run

```text
W Weight
```

e a concatenação preserva essa declaração antes de cada evento. O `HepMC3::ReaderAscii` interpreta a primeira declaração como `GenRunInfo` global e falha no segundo bloco com:

```text
ReaderAscii::parse_weight_values: The number of weights (0)
does not match the number weight names(1) in the GenRunInfo object
```

Portanto, o batching não deve depender inicialmente do HepMC3 agregado. Os arquivos individuais em

```text
EventGeneration/jobs/<powheg_request_id>/events.hepmc3
```

são válidos e devem ser a entrada canônica por sítio. Em paralelo, `_merge_hepmc` precisa ser corrigido e validado por leitura real com HepMC3, não apenas por inspeção textual.

## 3. Modelo físico correto

### 3.1 Separação entre escala global e escala local

Cada interação é definida globalmente por coordenadas Boyer–Lindquist-like

```text
(r_int, theta_int, phi_int)
```

e por um referencial ortonormal local associado à matéria. O Geant4 não deve transportar diretamente em coordenadas Kerr. Ele recebe o estado final H3-W10 no referencial local da matéria e resolve a cascata em uma aproximação tangente aproximadamente plana.

Para `M=3 M_sun`:

```text
r_g = 4.430009 km
patch half-size = 10 mm = 2.2573317e-6 r_g
```

Logo, cada patch é um inset local, e não um volume desenhado na escala do BH+toro.

### 3.2 Densidade usada

Com `density_source=dis_vertex_local`, a densidade vem exatamente do mesmo modelo compartilhado usado pelo DIS:

```text
rho(r,theta) = rho0
               exp[-0.5 ((r-r_peak)/Delta_r)^2]
               exp[-0.5 ((theta-pi/2)/sigma_theta)^2]
```

com corte radial duro em `[r_inner,r_outer]`. A configuração corrente é:

- `rho0 = 1e10 g/cm3`;
- `r_inner = 6 r_g`;
- `r_peak = 10 r_g`;
- `r_outer = 14 r_g`;
- `sigma_theta = 10 deg`;
- `Delta_r = (r_outer-r_inner)/2 = 4 r_g`.

Essas densidades são locais e reais **dentro do modelo analítico do HADROS3**. Elas não são perfis importados de GRMHD, nem incluem temperatura, ionização, degenerescência ou composição nuclear calculadas por uma equação de estado.

### 3.3 Validade da aproximação homogênea dentro de cada patch

Para o perfil analítico, uma estimativa da variação fracionária em um deslocamento local `h` é

```text
epsilon_rho ~= h |grad ln rho|.
```

Usando `h=10 mm` e os 36 sítios correntes:

- mediana estimada: `3.22e-6`;
- máximo estimado: `8.33e-6`.

Assim, congelar a densidade dentro de cada patch de 20 mm é numericamente consistente com o perfil analítico. O erro atual não é a homogeneidade de cada patch; é reutilizar um único patch para centros globais diferentes.

O aceite proposto é:

```text
epsilon_rho <= 1e-4
```

ou então subdividir/rejeitar o patch.

### 3.4 Material, composição e limites científicos

Com `material=HADROS3_H_HE`, o backend cria um `G4Material` H/He com:

- densidade igual à densidade do sítio;
- fração mássica de hidrogênio configurada, atualmente `0.75`;
- fração de hélio `0.25`.

O Geant4 descreve densidade, composição e propriedades macroscópicas através de `G4Material`; essas propriedades entram em comprimentos de radiação, livre caminho médio, `dE/dx` e tabelas de física. Materiais NIST (`G4_WATER`, `G4_Pb`, `G4_Galactic`) usam suas densidades internas e não devem ser anunciados como material local do toro.

Há uma limitação científica séria: `rho ~ 1e6--1e10 g/cm3` representa plasma astrofísico denso, enquanto o material H/He atual é uma mistura atômica macroscópica. O FTFP_BERT e a física EM padrão incluem efeitos materiais importantes, inclusive supressões LPM e dielétrica na física de bremsstrahlung de alta energia, mas isso não equivale a validar plasma degenerado, ionização, screening coletivo ou uma EOS astrofísica nessa faixa de densidade.

Portanto, a correção de batching torna a geometria e a densidade **internamente consistentes**, mas não encerra a validação de matéria astrofísica. O manifest deve publicar:

```text
material_model = HADROS3_H_HE_macroscopic_proxy_v1
thermodynamic_state_model = not_modeled
plasma_collective_effects_validation = not_established
```

### 3.5 Tamanho físico versus coluna atravessada

Um patch de 20 mm em `rho=4.0e9 g/cm3` representa coluna aproximada

```text
X = rho L ~= 8.0e9 g/cm2.
```

Isso é extremamente opaco. O tamanho de `10 mm` não deve permanecer apenas como escolha geométrica arbitrária. Devem coexistir dois modos explícitos:

1. `fixed_physical_half_size_mm`: útil para testes e comparação direta;
2. `target_column_depth_g_cm2`: deriva `L=X/rho` e é preferível para estudos de convergência/contensão da cascata.

O objetivo científico também deve ser declarado:

- **chuveiro local:** transportar até uma coluna definida e exportar o estado de fronteira;
- **propagação pelo toro inteiro:** exige uma cadeia de células/patches com densidade variável e acoplamento ao transporte global; não é resolvida por um único patch local.

## 4. Uso correto das APIs Geant4

### 4.1 Condições compartilhadas por um run

No modelo do Geant4, os eventos de um `G4Run` compartilham detector, geometria, materiais, regiões, cuts e lista física. `G4LogicalVolume` associa o sólido ao material. As tabelas físicas dependem dos pares material/cuts e são preparadas antes do loop de eventos.

Consequência: não se deve trocar arbitrariamente a densidade no meio de um `BeamOn(N)`.

É possível modificar geometria/material entre runs e notificar o `G4RunManager`, mas isso exige reconstrução/revalidação de geometria e, quando necessário, de tabelas físicas. Essa via é mais complexa e oferece pouco benefício antes de existir um baseline correto.

### 4.2 Lista física

O baseline permanece `FTFP_BERT`:

- Bertini cobre os hádrons leves em baixa energia;
- FTF cobre os mesmos hádrons até `100 TeV` na implementação descrita pelo guia;
- a lista usa `G4EmStandardPhysics`;
- é indicada para calorimetria, raios cósmicos e partículas energéticas, com ressalva oficial para colisões da ordem de `10 TeV` ou mais.

O teto conservador H3-W11 de `100000 GeV` deve continuar bloqueante até haver uma campanha externa por espécie e energia. A energia atual do neutrino, `~1e4 GeV`, não garante que toda aproximação do meio denso esteja validada.

### 4.3 Cuts

`production_cut_mm` é um threshold de produção em alcance, convertido em energia para cada `G4MaterialCutsCouple`; não é um tracking cut. Mudar a densidade muda a conversão material-dependente e pode mudar multiplicidade e deposição.

Cada sítio deve registrar:

- cut em alcance;
- thresholds de energia efetivos para gamma/e-/e+/proton;
- lista física e versão Geant4;
- material, densidade e composição.

É obrigatório repetir estudos de convergência em cuts para densidades representativas baixa, mediana e alta.

### 4.4 Scoring e passos

O backend atual usa `G4UserSteppingAction`, uma escolha válida para registrar steps e fronteiras. Para execução paralela interna, entretanto, as user actions são thread-local e o acumulador atual não é thread-safe.

No baseline por subprocesso:

- cada backend continua serial;
- não há estado compartilhado dentro do Geant4;
- `SteppingAction` pode permanecer;
- o Python agrega somente depois que o processo termina.

Como otimização posterior, um `G4VSensitiveDetector` no patch e `G4Run::Merge`/`G4Accumulable` podem reduzir contenção e organizar scoring em MT.

### 4.5 Seeds e reprodutibilidade

O seed não pode depender da posição do job na fila nem do worker que terminou primeiro. A política proposta é

```text
seed64 = hash(
    geant4_base_seed,
    event_generation_event_id,
    interaction_id,
    replica_index,
    geant4_configuration_sha256
)
```

reduzido ao domínio aceito pelo engine configurado. Para várias réplicas no mesmo processo, o seed deve ser reinstalado no início de cada evento, ou o backend deve implementar uma estratégia explícita de seeds por evento.

O Geant4 documenta seeds associados a eventos como requisito de reprodutibilidade independente do número de threads. O baseline serial por sítio deve alcançar o mesmo objetivo independentemente da ordem/concorrência externa.

## 5. Alternativas arquiteturais

| Estratégia | Correção física | Complexidade | Paralelismo | Isolamento de falha | Decisão |
|---|---:|---:|---:|---:|---|
| um job único, uma densidade | não | baixa | simples | baixa | rejeitar |
| trocar material entre eventos no mesmo run | não suportado como desenho simples | alta | ruim | baixa | rejeitar |
| vários runs no mesmo processo, reinitializando material | possível | alta | serial/complexo | baixa | estudar depois |
| vários patches separados numa geometria única | possível com cuidados | muito alta | MT interno | baixa | otimização futura |
| um subprocesso serial por sítio | sim | moderada | excelente externamente | alta | **baseline recomendado** |

### 5.1 Por que subprocessos são o baseline

- cada processo possui seu próprio singleton `G4RunManager`;
- cada processo possui apenas um material local;
- tabelas material/cuts não são compartilhadas incorretamente;
- falha/crash de um sítio não corrompe os demais;
- logs, seeds e produtos ficam naturalmente particionados;
- o backend C++ atual quase não precisa mudar;
- o paralelismo externo apresentou boa eficiência real.

### 5.2 Otimização futura: geometria multipatch

Uma geometria artificial poderia colocar patches não sobrepostos, cada um com seu material, e gerar cada evento dentro do patch correspondente. Isso permitiria `G4MTRunManager` e inicialização única. Entretanto:

- seriam criados muitos `G4MaterialCutsCouple` e tabelas;
- as coordenadas locais precisariam ser deslocadas e restauradas;
- tracks escapantes teriam de ser encerradas na fronteira para não entrar em outro patch;
- a saída atual não é thread-safe;
- a memória e o tempo de inicialização podem crescer com o número de materiais;
- um erro de geometria afetaria todo o lote.

Ela só deve avançar se benchmarks com centenas/milhares de sítios mostrarem que a inicialização por processo domina o custo.

## 6. Arquitetura proposta

### 6.1 Identidade do sítio

Definir:

```text
site_key = hash(
    interaction_id,
    material_model,
    density_g_cm3,
    composition,
    patch_geometry,
    production_cuts,
    physics_list,
    local_frame_id
)
```

Eventos com o mesmo `site_key` podem compartilhar um job. Densidade numericamente parecida não autoriza agrupamento; o agrupamento é por identidade física/proveniência, não por arredondamento.

### 6.2 Plano de jobs

Novo produto antes de executar:

```text
GEANT4/geant4_site_jobs_plan.jsonl
```

Cada linha contém:

- `site_job_id`;
- `site_key`;
- `interaction_id`;
- lista de `event_generation_event_id`;
- `powheg_request_id`;
- HepMC3 de entrada e SHA-256;
- `r,theta,phi` e posição cartesiana global;
- ID e componentes da tétrada local;
- densidade e origem da densidade;
- composição;
- dimensões do patch e coluna;
- seed de cada evento;
- custo estimado;
- status inicial `planned`.

### 6.3 Entrada por job

O planner deve ler `EventGeneration/event_generation_manifest.json` e `event_generation_jobs.jsonl`, nunca inferir caminhos apenas pelo índice.

Para cada request:

```text
EventGeneration/jobs/<powheg_request_id>/events.hepmc3
```

é a entrada preferida. Antes de executar:

1. abrir com `HepMC3::ReaderAscii` em modo `import_check`;
2. verificar cardinalidade;
3. comparar os IDs com `events_summary.jsonl`;
4. verificar unidades `GEV MM`;
5. verificar SHA-256 e imutabilidade.

### 6.4 Diretórios

```text
GEANT4/
  site_jobs/
    H3G4SITE-000001/
      site_job_manifest.json
      geant4_import_report.json
      geant4_backend_summary.json
      events_summary.jsonl
      escaped_particles.jsonl
      steps.jsonl
      stdout.log
      stderr.log
    H3G4SITE-000002/
      ...
  geant4_site_jobs_plan.jsonl
  geant4_site_jobs_summary.jsonl
  geant4_events_summary.jsonl
  geant4_escaped_particles.jsonl
  geant4_steps.jsonl
  geant4_sites.json
  geant4_manifest.json
  geant4_validation_report.json
  geant4_summary.json
```

### 6.5 Scheduler

Implementar scheduler Python com subprocessos seriais:

```text
workers = min(
    configured_site_workers,
    number_of_sites,
    cpu_budget,
    memory_budget
)
```

Baseline recomendado para a máquina auditada:

```text
geant4.site_workers = 4
geant4.backend_threads_per_site = 1
```

O scheduler deve:

- lançar no máximo `workers` processos;
- guardar PID e tempos;
- capturar stdout/stderr separados;
- suportar cancelamento com terminate, grace period e kill;
- não publicar produtos parciais como resultado final;
- preservar relatório de falha por sítio;
- ordenar a agregação pelo índice global do evento, nunca pela ordem de término.

### 6.6 Agregação

Todos os IDs locais do backend devem ser namespaced:

```text
site_job_id
site_event_index
global_geant4_event_id
event_generation_event_id
interaction_id
```

Na agregação:

- `global_geant4_event_id` é atribuído deterministicamente pelo plano;
- `track_id` continua local ao evento e deve ser acompanhado dos IDs globais;
- paths são reescritos para a árvore final;
- hashes de cada job entram em uma Merkle-like list ordenada no manifest;
- qualquer duplicata ou evento ausente falha a validação.

### 6.7 Política de falha

Default científico:

```text
site_failure_policy = fail_stage
```

Se um sítio falhar:

- jobs em execução podem terminar ou ser cancelados conforme configuração;
- o staging não substitui o resultado anterior válido;
- um `geant4_failed_site_jobs.jsonl` é preservado fora do produto publicado ou em área diagnóstica;
- a UI mostra qual sítio falhou, densidade, energia máxima, exit code e log.

Um futuro modo `publish_partial_with_explicit_missing_weight` só deve existir depois de definir como renormalizar pesos e taxas; não pode ser um default operacional.

## 7. Estudo de velocidade de execução

### 7.1 Ambiente medido

- CPU visível: 12 processadores lógicos;
- RAM: 15 GiB, aproximadamente 8.1 GiB disponíveis durante a medição;
- Geant4: 11.4.2;
- HepMC3: 3.3.1;
- backend: serial;
- física: FTFP_BERT;
- material: `HADROS3_H_HE`;
- energia do evento: aproximadamente `1e4 GeV`;
- patch: 10 mm de half-size;
- medição em `/tmp`, sem geração de plots Python.

### 7.2 Inicialização e custo físico

Para `H3PWHG-000001`:

| caso | wall [s] | steps | escapes |
|---|---:|---:|---:|
| `import_check` | 0.0365 | — | — |
| vácuo | 0.3704 | 66 | 33 |
| H/He, `rho=4.015e9 g/cm3` | 2.0144 | 3,170,946 | 5,377 |

Interpretação:

- parse/audit isolado é barato;
- inicialização Geant4 + física + geometria aparece no baseline de vácuo, cerca de `0.37 s`;
- a cascata densa domina o restante do tempo;
- para essa amostra, eliminar toda a inicialização economizaria no máximo cerca de 18% por job; não justifica de início uma arquitetura multipatch muito mais complexa.

### 7.3 Quatro sítios: sequencial versus paralelo externo

Os quatro primeiros sítios foram executados individualmente com seus próprios HepMC3 e densidades.

| sítio | densidade [g/cm³] | steps | wall sequencial individual [s] |
|---|---:|---:|---:|
| H3DIS-000020 | `4.015e9` | 3,170,946 | 2.002 |
| H3DIS-000035 | `1.358e9` | 3,255,312 | 2.108 |
| H3DIS-000022 | `9.812e9` | 4,080,227 | 2.526 |
| H3DIS-000007 | `6.273e9` | 3,823,702 | 2.450 |

Resultados do lote:

```text
sequencial: 9.0868 s
4 processos concorrentes: 2.5141 s
speedup: 3.6143x
eficiência paralela: 90.4%
```

O RSS máximo medido para um processo denso foi aproximadamente `58 MiB`. Isso favorece quatro workers nessa máquina. O número ótimo ainda deve ser medido para 1, 2, 4, 6 e 8 workers, porque tabelas, cache de páginas, I/O e memória disponível mudam com o lote.

Estimativa não vinculante para 36 sítios semelhantes:

```text
sequencial: ~82 s
4 workers: ~23 s
```

A UI deve mostrar estimativa com intervalo, não um número exato, pois multiplicidade e número de steps variam por evento.

### 7.4 Custo do registro de steps

No mesmo evento denso:

| cap de steps | wall [s] | arquivo JSONL |
|---:|---:|---:|
| 100 | 1.991 | 78,649 bytes |
| 1,000 | 2.000 | 791,604 bytes |
| 50,000 | 2.251 | 39,926,497 bytes |

Registrar 50 mil steps aumentou o tempo do backend em aproximadamente 13% e gerou cerca de 40 MB para um único sítio. Para 36 sítios, o mesmo cap pode produzir aproximadamente 1.4 GB antes de agregação e viewers.

O cap atual registra os primeiros `N` steps, o que introduz viés temporal/geracional. A proposta deve substituí-lo por saídas em camadas:

1. **sempre completo:** resumo por evento, energia, escapes, contagens por processo e espécie;
2. **sempre preservado:** fronteiras, vértices de alta deposição e linhagens principais;
3. **amostra determinística:** reservoir/estratificação de steps por processo, track e geração;
4. **opt-in:** raw steps completos para sítios escolhidos pelo usuário.

Novas configurações:

```text
max_recorded_steps_per_site
max_recorded_steps_total
viewer_steps_per_site
raw_steps_site_allowlist
step_sampling_policy = deterministic_stratified_v1
```

### 7.5 Modelo de tempo para o scheduler

Para cada sítio `i`:

```text
T_i ~= T_init(material_i,cuts)
      + N_steps_i / R_steps
      + T_IO_i.
```

Para `W` workers:

```text
T_batch >= max(sum(T_i)/W, max(T_i)).
```

O scheduler deve ordenar opcionalmente por custo estimado decrescente (`longest processing time first`) para reduzir a cauda. Estimadores possíveis:

- multiplicidade primária H3-W10;
- energia total inicial;
- coluna `rho L`;
- espécie de maior energia;
- histórico por bins de energia/densidade.

O estimador nunca deve alterar seeds ou resultados, apenas a ordem de lançamento.

## 8. Plano de implementação

### Fase B0 — congelar contratos e fixtures

1. Criar fixtures com:
   - dois eventos no mesmo sítio;
   - dois sítios com densidades distintas;
   - 36 jobs pequenos;
   - um job acima do teto físico;
   - um HepMC3 agregado inválido com headers repetidos.
2. Registrar hashes e resultados atuais.
3. Documentar `site_key`, IDs globais e unidade/frame.

**Aceite B0:** schemas validam os fixtures e reproduzem o erro atual de densidade e o erro atual de merge HepMC3.

### Fase B1 — corrigir o contrato HepMC3

1. Tornar `EventGeneration/jobs/.../events.hepmc3` entrada canônica por job.
2. Corrigir `_merge_hepmc` usando Reader/Writer HepMC3 ou writer único.
3. Renumerar eventos globalmente sem alterar IDs HADROS3 externos.
4. Validar o agregado reabrindo-o com HepMC3.

**Aceite B1:** 36 eventos escritos, 36 lidos, pesos/run info válidos e quatro-momento inalterado.

### Fase B2 — planner por sítio

1. Juntar Event Generation, POWHEG request e DIS por IDs.
2. Construir `site_key` e plano JSONL.
3. Agrupar somente eventos fisicamente equivalentes.
4. Calcular densidade, coluna, gradiente e seed.
5. Recusar metadados ausentes antes de lançar Geant4.

**Aceite B2:** os 36 eventos correntes geram 36 sítios; réplicas do mesmo request geram um sítio com múltiplos eventos.

### Fase B3 — executor serial por sítio

1. Extrair a execução atual para `run_geant4_site_job`.
2. Criar diretório e manifest por sítio.
3. Passar densidade/material específicos.
4. Preservar o guard homogêneo dentro do job.
5. Implementar seed por evento.

**Aceite B3:** dois sítios de densidades diferentes completam independentemente e cada manifest contém a densidade correta.

### Fase B4 — scheduler paralelo e cancelamento

1. Implementar fila com limite de workers.
2. Adicionar PID tracking, timeout, cancelamento e limpeza segura.
3. Implementar estimativa de custo e ordenação LPT opcional.
4. Impedir oversubscription (`site_workers * backend_threads_per_site`).

**Aceite B4:** resultados byte a byte iguais com 1 e 4 workers; cancelamento não publica lote incompleto.

### Fase B5 — agregador determinístico

1. Reescrever IDs globais.
2. Mesclar summaries, escapes e steps amostrados.
3. Validar cardinalidade e unicidade.
4. Publicar manifest de hashes por sítio.
5. Fazer rename atômico do staging.

**Aceite B5:** ordem dos arquivos finais independe da ordem de término dos jobs.

### Fase B6 — visualização e UI

1. Mostrar progresso `completed/total` e job ativo.
2. Mostrar densidade/material por marcador macro.
3. Abrir viewer local do sítio selecionado.
4. Indicar steps totais, registrados e amostrados.
5. Permitir retry apenas de sítios falhos sem reutilizar resultado stale.

**Aceite B6:** 36 marcadores correspondem aos 36 `interaction_id` e cada link abre o volume correto.

### Fase B7 — desempenho e física

1. Benchmark 1/2/4/6/8 workers.
2. Benchmark cuts, coluna e cap de steps.
3. Comparar distribuições com execução serial.
4. Validar aproximação homogênea.
5. Auditar validade da matéria H/He densa e documentar envelope.

**Aceite B7:** configuração default escolhida por evidência, sem alterar observáveis além das tolerâncias estatísticas/numerárias.

## 9. Campanha de testes numéricos

### T-B01 — isolamento de densidade

- dois sítios, densidades `rho1 != rho2`;
- manifests devem registrar exatamente `rho1` e `rho2`;
- nenhuma linha do evento 1 pode conter `site_job_id` do evento 2.

### T-B02 — equivalência single-site

- comparar H3-W11 v1 de um evento com novo executor de um sítio;
- mesmo seed;
- summaries e escapes byte a byte idênticos.

### T-B03 — independência da concorrência

- executar mesmo lote com 1 e 4 workers;
- produtos agregados canônicos idênticos;
- logs/tempos podem diferir e ficam fora do hash científico.

### T-B04 — cardinalidade

```text
N_input_H3W10 = sum(N_events_site_job)
              = N_events_transported
              = N_event_summaries.
```

### T-B05 — densidade e gradiente

- recomputar `rho(r,theta)` no teste;
- erro relativo `<1e-12`;
- `epsilon_rho<=1e-4` ou status explícito de patch inválido.

### T-B06 — frame

- base local ortonormal dentro da tolerância;
- transformações local→global→local fecham;
- momento escapante conserva norma e orientação.

### T-B07 — energia por sítio

- aplicar ledger atual por evento;
- validar residual segundo modo vácuo/material;
- não somar trocas de massa do meio entre sítios antes da validação individual.

### T-B08 — cuts

- repetir em `0.2`, `0.1`, `0.05 mm` ou escala fisicamente adequada;
- comparar energia depositada, multiplicidade, escapes e espectros;
- definir tolerâncias antes de aceitar produção.

### T-B09 — coluna/tamanho

- variar coluna por fatores 0.5, 1, 2;
- verificar convergência de observáveis de interesse ou documentar dependência.

### T-B10 — falha parcial

- inserir espécie/energia inválida em um sítio;
- lote final não é publicado;
- sítio falho e causa são identificados;
- resultados anteriores válidos permanecem intactos.

### T-B11 — merge HepMC3

- escrever vários jobs com run info;
- ler agregado com `HepMC3::ReaderAscii`;
- pesos, unidades, eventos e partículas preservados.

### T-B12 — limites de disco

- atingir cap por sítio e cap global;
- summaries completos;
- sampling determinístico;
- viewer abre sem incorporar milhões de registros.

### T-B13 — benchmark

- medir wall, CPU, RSS, bytes e steps/s;
- registrar ambiente e carga;
- aceitar paralelismo somente se resultados científicos forem invariantes.

## 10. Mudanças de configuração propostas

```json
{
  "geant4": {
    "execution_model": "per_site_subprocess",
    "site_workers": 4,
    "backend_threads_per_site": 1,
    "site_grouping_policy": "exact_site_key",
    "site_failure_policy": "fail_stage",
    "patch_size_mode": "fixed_physical_half_size_mm",
    "target_column_depth_g_cm2": null,
    "max_recorded_steps_per_site": 5000,
    "max_recorded_steps_total": 100000,
    "viewer_steps_per_site": 2000,
    "step_sampling_policy": "deterministic_stratified_v1",
    "raw_steps_site_allowlist": []
  }
}
```

Os números de steps são propostas iniciais e devem ser ajustados pela campanha de desempenho/disco.

## 11. Critérios de conclusão

A correção só pode receber `COMPLLETO` quando:

1. eventos de densidades diferentes executarem sem densidade compartilhada;
2. cada resultado conservar `interaction_id`, frame, material e densidade;
3. o HepMC3 agregado for legível pelo HepMC3 real;
4. execução serial e paralela forem cientificamente idênticas;
5. falha parcial não publicar resultado incompleto;
6. UI mostrar todos os sítios e volumes locais corretos;
7. custos de CPU, RAM e disco forem limitados e registrados;
8. cuts e coluna tiverem estudo de convergência;
9. limitações da matéria H/He densa estiverem explícitas no manifest e na documentação.

## 12. Referências primárias

- [Geant4 11.4 — definição de materiais e densidade](https://geant4.web.cern.ch/documentation/dev/bfad_html/ForApplicationDevelopers/GettingStarted/materialDef.html)
- [Geant4 11.4 — logical volumes e associação com material](https://geant4.web.cern.ch/documentation/dev/bfad_html/ForApplicationDevelopers/Detector/Geometry/geomLogical.html)
- [Geant4 11.4 — ciclo de runs e `G4RunManager`](https://geant4.web.cern.ch/documentation/dev/bfad_html/ForApplicationDevelopers/Fundamentals/run.html)
- [Geant4 11.4 — Book for Application Developers](https://geant4.web.cern.ch/documentation/dev/bfad_pdf/BookForApplicationDevelopers.pdf)
- [Geant4 11.4 — paralelismo event-level e seeds](https://geant4.web.cern.ch/documentation/dev/bftd_html/ForToolkitDeveloper/OOAnalysisDesign/Multithreading/mt.html)
- [Geant4 11.4 — physics tables e `G4MaterialCutsCouple`](https://geant4.web.cern.ch/documentation/dev/bfad_html/ForApplicationDevelopers/TrackingAndPhysics/physicsTable.html)
- [Geant4 11.4 — production threshold versus tracking cut](https://geant4.web.cern.ch/documentation/dev/bfad_html/ForApplicationDevelopers/TrackingAndPhysics/thresholdVScut.html)
- [Geant4 11.4 — cuts por região](https://geant4.web.cern.ch/documentation/dev/bfad_html/ForApplicationDevelopers/TrackingAndPhysics/cutsPerRegion.html)
- [Geant4 11.4 — FTFP_BERT](https://geant4.web.cern.ch/documentation/dev/plg_html/PhysicsListGuide/reference_PL/FTFP_BERT.html)
- [Geant4 11.4 — LPM e supressão dielétrica](https://geant4.web.cern.ch/documentation/dev/prm_html/PhysicsReferenceManual/electromagnetic/electron_incident/bremsstrahlung/ebrem.html)
- [Geant4 11.4 — sensitive detectors e hits](https://geant4.web.cern.ch/documentation/dev/bfad_html/ForApplicationDevelopers/Detector/hit.html)
- [Geant4 11.4 — eventos, trajetórias e hits](https://geant4.web.cern.ch/documentation/dev/bfad_html/ForApplicationDevelopers/Fundamentals/event.html)

## 13. Decisão proposta

Adotar imediatamente como alvo de implementação:

```text
H3-W11 per-site subprocess batching
serial GEANT4 backend per site
external bounded parallelism
exact site/material identity grouping
deterministic aggregation
```

Manter como pesquisa posterior, não como requisito da primeira correção:

```text
single-process multipatch geometry
G4MTRunManager across heterogeneous sites
dynamic material reinitialization between runs
```

Essa ordem entrega primeiro correção física e auditabilidade, aproveita o paralelismo já medido e evita acoplar a correção urgente a uma reescrita multithread do backend Geant4.
