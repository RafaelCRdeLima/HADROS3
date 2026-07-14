# HADROS3 — Plano detalhado de implementação do GEANT4

**Status:** campanha em execução; backend suportado implementado e testado, produção UHE bloqueada pelo domínio físico  
**Data:** 2026-07-13  
**Estágio:** H3-W11  
**Entrada oficial:** H3-W10 Event Generation, em HepMC3  
**Saída seguinte:** H3-W12 Photon Transport

## 1. Decisão executiva

O H3-W11 deve transportar, em matéria, as partículas finais geradas pelo PYTHIA 8. A implementação recomendada é um executável C++17/20 baseado no **Geant4 11.4.2**, ligado ao HepMC3 já usado pelo projeto, com uma camada Python para configuração, execução, publicação atômica, validação, gráficos e integração com `hadros-web`.

```text
H3-W10 Event Generation
  event_generation_events.hepmc3
  metadados dos eventos, pesos e tétrada local
                  |
                  v
H3-W11 GEANT4
  transporte em um volume material local ao vértice DIS
  deposição de energia + histórico de processos + partículas escapantes
                  |
                  v
H3-W12 Photon Transport
  propagação em Kerr das partículas/fótons que deixaram o volume local
```

A primeira implementação física não deve tentar representar todo o toro em coordenadas de Boyer--Lindquist. Cada evento deve ser transportado em um **patch material local**, construído no referencial ortonormal da matéria no ponto da interação. Na primeira versão, densidade e composição são constantes dentro do patch e iguais às do vértice. Uma geometria estratificada poderá ser adicionada depois que o contrato homogêneo estiver validado.

Geant4 não será usado para:

- repetir o processo duro, o shower ou a hadronização;
- propagar geodésicas na métrica de Kerr;
- escolher imagens gravitacionais ou aplicar pesos do observador;
- transformar deposição local diretamente em espectro observado;
- simular um detector de laboratório, salvo em fixtures diagnósticos;
- extrapolar modelos hadrônicos para energias além do domínio documentado.

O último item é bloqueante. O evento atualmente produzido pelo H3-W10 contém, por exemplo, elétrons e hádrons com energias de milhões a centenas de milhões de GeV. A documentação da lista `FTFP_BERT` descreve o Fritiof até **100 TeV** para os hádrons cobertos. Portanto, "o programa rodou" não será critério de validade em UHE: a primeira versão deverá recusar, segregar ou encaminhar explicitamente qualquer partícula fora do envelope validado.

## 2. Estado atual auditado

| Componente | Estado em 2026-07-13 | Consequência para H3-W11 |
|---|---|---|
| `hadros3/config.py` | `geant4.mode=placeholder_disabled` e `planned_model=detector_transport_future` | Não há configuração executável e o nome do modelo não representa o patch astrofísico pretendido |
| `presets/hadros_web/default_config.json` | Repete o placeholder | O default deve continuar desabilitado, mas com opções reais após a implementação |
| `hadros_web.py` | Status GEANT4 sempre `pending`; renderer genérico | Faltam estado, botão, endpoint, diagnósticos e produtos |
| H3-W10 | Implementado; produz HepMC3, JSONL, manifest e relatório de validação | Existe uma entrada estável e auditável |
| HepMC3 | 3.3.1 instalado no ambiente `dis`; arquivo declara 3.03.01 | O adaptador pode ser implementado em C++ sem conversão intermediária |
| PYTHIA | 8.312 instalado e usado pelo H3-W10 | Não será ligado novamente dentro do H3-W11 |
| Geant4 | Não apareceu entre os pacotes instalados no ambiente `dis` | Bootstrap e validação de datasets são entregas obrigatórias |
| Modelo do toro | Define geometria, perfil e normalização de densidade; não define composição química/nuclear suficiente | Não é possível construir material físico sem adicionar um contrato de composição |
| Teoria vigente | Marca H3-W11 como planejado e não implementado | Só deve mudar após toda a campanha de aceite |

O arquivo oficial atual `EventGeneration/event_generation_events.hepmc3` já declara:

```text
HepMC::Version 3.03.01
U GEV MM
```

O manifest registra ainda:

```text
generator_frame = local_matter_tetrad
momentum_unit = GeV
length_unit = mm
stage = H3-W10
```

Esses campos devem ser pré-condições verificadas, não suposições.

## 3. Fronteira física

### 3.1 Objeto físico transportado

Cada evento H3-W10 aceito vira exatamente um `G4Event`. As partículas de estado final selecionadas pelo contrato HepMC3 viram primárias no mesmo `G4PrimaryVertex` local. O H3-W11 acompanha:

- processos eletromagnéticos e hadrônicos habilitados;
- criação e término de secundários;
- energia depositada no material;
- comprimentos de trajetória e tempos locais;
- partículas que cruzam a fronteira externa do patch;
- partículas recusadas por espécie ou energia fora de domínio;
- um balanço energético auditável por evento.

### 3.2 Referencial e coordenadas

O sistema de coordenadas do Geant4 será cartesiano e ortonormal:

```text
(x_G4, y_G4, z_G4) = (e_(1), e_(2), e_(3)) da tétrada local da matéria
t_G4              = tempo próprio/local do patch
origem             = vértice DIS
```

A transformação entre a tétrada local e as coordenadas globais de Kerr deve viajar como metadado. Ela não será aplicada durante os steps do Geant4 na primeira versão. Isso evita representar gravidade curvada como uma força efetiva não validada.

Convenções obrigatórias:

- momento e energia na entrada: GeV;
- posição HepMC3: mm;
- tempo HepMC3: `mm/c`, convertido explicitamente para unidade interna;
- densidade de configuração: `g/cm3`, convertida com constantes do Geant4;
- eixos com orientação e determinante da tétrada testados;
- nenhum flip, troca de eixos ou rotação implícita para fins de visualização.

### 3.3 Escala do patch

O tamanho físico do patch não pode ser escolhido por conveniência gráfica. Ele deve ser derivado de uma das políticas configuráveis:

1. `fixed_local_length`: comprimento físico explícito em cm ou m;
2. `fraction_of_density_scale_height`: fração da escala local de densidade;
3. futuramente, `distance_to_torus_boundary`: distância até a superfície do toro na direção local.

O MVP usará `fixed_local_length` e salvará o tamanho no manifest. A versão de produção só será aceita depois de um teste de convergência em espessura/coluna de matéria.

### 3.4 Matéria

A densidade não define sozinha um `G4Material`. O contrato deverá separar:

- `density_g_cm3`;
- frações mássicas de H, He e metais ou uma composição nuclear explícita;
- estado termodinâmico usado apenas quando relevante à física selecionada;
- energia de ionização ou propriedades ópticas somente quando justificadas;
- origem e versão da prescrição astrofísica.

Não será adotada silenciosamente "água", "silício" ou "hidrogênio puro". Antes de produção, a teoria deverá escolher e justificar uma composição do toro. Até lá, `hydrogen_fixture`, `lead_fixture` e materiais NIST só poderão aparecer em testes marcados como diagnósticos.

### 3.5 Campos e gravidade

O baseline não inclui campo magnético, elétrico nem gravidade dentro do patch. Isso será registrado como `field_model=none_local_inertial_patch`. Um campo magnético poderá ser introduzido em fase posterior com:

- componentes na tétrada local;
- integrador e tolerâncias declarados;
- convergência em passo;
- teste do raio de Larmor;
- separação explícita entre curvatura magnética local e geodésica global.

## 4. Restrição crítica de domínio UHE

### 4.1 Problema

A lista de referência candidata, `FTFP_BERT`, combina Bertini em baixas energias com FTF em altas energias. O guia oficial documenta intervalos de transição e indica FTF até 100 TeV para vários hádrons. O H3-W10 já produziu partículas acima de `10^8 GeV`, aproximadamente quatro ordens de grandeza acima desse teto.

Geant4 pode aceitar um `double` com essa energia, mas isso não valida seção de choque, multiplicidade, fragmentação ou inelasticidade. Extrapolação silenciosa seria um erro físico grave.

### 4.2 Política obrigatória

O adaptador deve manter uma tabela versionada:

```text
(PDG, processo/lista física) -> [E_min_validada, E_max_validada, fonte]
```

Antes de `BeamOn`, cada primária será classificada como:

- `supported`: pode entrar no transporte validado;
- `pass_through`: espécie deliberadamente não interagente, preservada até a fronteira;
- `unsupported_species`: PDG sem definição ou política;
- `unsupported_energy`: fora do envelope aceito;
- `requires_external_uhe_model`: exige outro backend ou parametrização validada.

O default de produção será `unsupported_policy=fail_stage`. O modo diagnóstico poderá usar `quarantine_and_report`, mas nunca contará essas partículas como transportadas. Não haverá `clip_energy`.

### 4.3 Estratégia em duas faixas

O desenvolvimento deve separar:

- **faixa A — Geant4 validável:** fixtures e eventos com espécies/energias dentro do domínio documentado; permite validar todo o software e a física local suportada;
- **faixa B — produção HADROS3 UHE:** bloqueada até existir uma decisão científica documentada sobre extensão do modelo, acoplamento a gerador UHE externo ou aproximação de transporte contínuo.

Concluir a faixa A não autoriza rotular a faixa B como completa.

### 4.4 Neutrinos e fótons

- Neutrinos finais serão `pass_through` por default. Não se presumirá que `FTFP_BERT` modele interação UHE de neutrinos.
- Fótons gama seguem a física EM dentro do envelope documentado e devem ser exportados ao cruzar a fronteira.
- Fótons ópticos Geant4 ficam desabilitados no MVP. Cintilação/Cherenkov geraria outro problema físico e computacional e não é sinônimo do H3-W12.

## 5. Entradas oficiais e pré-condições

### 5.1 Arquivos consumidos

O estágio deve consumir somente produtos publicados pelo H3-W10:

- `EventGeneration/event_generation_events.hepmc3`;
- `EventGeneration/event_generation_manifest.json`;
- `EventGeneration/event_generation_events_summary.jsonl`;
- `EventGeneration/event_generation_final_particles.jsonl`, apenas para validação cruzada;
- metadados upstream referenciados no manifest, incluindo evento POWHEG, interação DIS, posição e tétrada.

### 5.2 Verificações bloqueantes

A execução deve falhar antes de construir o `G4RunManager` quando:

- o resumo H3-W10 não tiver `status=ok`;
- o resultado estiver stale em relação à configuração ou aos hashes upstream;
- o HepMC3 não terminar corretamente ou não puder ser lido;
- unidades não forem exatamente reconhecidas e convertíveis;
- `generator_frame` não for `local_matter_tetrad`;
- cardinalidade e IDs divergirem entre HepMC3, resumo e sidecar;
- faltar densidade, composição, tamanho do patch ou tétrada;
- um PDG final não tiver política explícita;
- alguma primária estiver fora do domínio físico configurado;
- Geant4, HepMC3 ou datasets requeridos estiverem ausentes/incompatíveis;
- o binário tiver sido compilado contra versões diferentes das registradas sem nova validação.

Não deve existir fallback para evento sintético, material default ou geometria de detector.

## 6. Arquitetura de software

### 6.1 Camadas

```text
hadros-web / CLI / Makefile
            |
            v
hadros3/geant4_transport.py
  contrato, hashes, jobs, staging, agregação, plots e validação
            |
            v
bin/hadros3_geant4_transport
  HepMC3 -> G4PrimaryVertex -> Geant4 -> JSONL/CSV por job
            |
            v
GEANT4/ produtos oficiais e fronteira H3-W12
```

Python não deve reimplementar steps ou física. C++ não deve decidir layout de publicação, HTML nem estado da pipeline.

### 6.2 Estrutura proposta

```text
cpp/
  apps/hadros3_geant4_transport.cpp
  include/hadros3/geant4/
    Config.hh
    DetectorConstruction.hh
    MaterialFactory.hh
    HepMC3PrimaryGenerator.hh
    PhysicsListFactory.hh
    EventAction.hh
    RunAction.hh
    TrackingAction.hh
    SteppingAction.hh
    OutputWriter.hh
    DomainGuard.hh
  src/geant4/
    ...implementações...

cmake/
  FindHADROS3Geant4.cmake

hadros3/
  geant4_transport.py

schemas/
  geant4_transport_contract.schema.json

scripts/geant4/
  bootstrap_geant4.py
  inspect_geant4_environment.py

tests/
  test_geant4_transport.py
  fixtures/geant4/
    one_muon_vacuum.hepmc3
    mono_gamma_slab.hepmc3
    mono_electron_slab.hepmc3
    mono_proton_slab.hepmc3
    unsupported_uhe_pion.hepmc3

docs/
  GEANT4_BOOTSTRAP.md
  HADROS3_GEANT4_Validation_Report.md
```

### 6.3 Build

Geant4 deve ser encontrado via CMake:

```cmake
find_package(Geant4 11.4.2 REQUIRED)
find_package(HepMC3 3.3 REQUIRED)
```

O Makefile principal oferecerá wrappers, não uma segunda lógica de build:

```text
make geant4-bootstrap
make geant4-build
make geant4-import-check
make geant4-smoke
make geant4-validate
```

O executável não dependerá de Qt ou OpenGL em produção headless. Visualizações oficiais serão geradas a partir das saídas normalizadas; uma build com visualização Geant4 poderá ser opcional para depuração.

### 6.4 Instalação reproduzível

O baseline será a versão estável **11.4.2**, não a beta 11.5. O bootstrap deve:

1. procurar um pacote compatível no ambiente `dis`;
2. registrar versão, compilador, flags, prefixo e hash do binário;
3. verificar todos os datasets exigidos e suas variáveis de ambiente;
4. compilar um programa mínimo que inicializa a lista física;
5. executar uma partícula em vácuo;
6. emitir `geant4_environment_manifest.json`;
7. falhar se datasets forem baixados ou encontrados em versões diferentes das esperadas.

Não basta encontrar `geant4-config`: a aplicação usa `Geant4Config.cmake` como contrato principal.

## 7. Adaptador HepMC3 -> Geant4

### 7.1 Leitura

Será usado `HepMC3::ReaderAscii`. O exemplo legado HepMC do Geant4 não deve ser copiado sem auditoria, pois exemplos antigos podem usar a API HepMC2.

Para cada evento:

1. ler o evento completo;
2. validar unidades;
3. recuperar ID HADROS3 por sidecar/atributo;
4. selecionar apenas partículas finais segundo política versionada;
5. validar `E^2-p^2=m^2`, energia positiva e valores finitos;
6. resolver cada PDG em `G4ParticleTable`;
7. criar uma primária com quatro-momento e massa consistentes;
8. criar o vértice na origem local, preservando deslocamento/tempo se o contrato vier a permiti-los;
9. anexar metadados de rastreabilidade fora das estruturas descartáveis do Geant4.

### 7.2 Definição de partícula final

O status PYTHIA não deve ser assumido como `status==1` apenas. O H3-W10 atual possui finais com códigos como 23 e 83 no JSONL. O contrato canônico deve usar a lista efetivamente marcada como final pelo backend H3-W10 e validar ausência de filhas, estabilidade/política e correspondência com o HepMC3.

### 7.3 IDs e linhagem

IDs mínimos preservados:

- `event_generation_event_id`;
- `powheg_request_id`;
- `lhe_event_index`;
- `geant4_event_id`;
- `primary_particle_index` e PDG;
- `track_id`, `parent_track_id` e geração;
- processo criador e processo final;
- ID da interação DIS/Observer Bridge quando disponível.

Um ID Geant4 numérico sozinho não é identidade científica estável.

### 7.4 Pesos

O transporte analógico não altera automaticamente pesos do gerador. Devem permanecer em namespaces separados:

```text
generator_weights.*
dis_sampling_weight
observer_weights.*
geant4_statistical_weight
```

No MVP, `geant4_statistical_weight=1`. Técnicas futuras de splitting ou Russian roulette só poderão ser habilitadas com propagação explícita do peso por track e testes de estimadores não viesados.

## 8. Geometria e materiais

### 8.1 Geometrias

| Geometria | Uso | Pode produzir ciência? |
|---|---|---|
| `vacuum_world_fixture` | identidade, unidades, fronteira e determinismo | não |
| `nist_slab_fixture` | atenuação e dE/dx contra referências | não |
| `local_homogeneous_patch_v1` | primeira geometria astrofísica | sim, após composição e domínio validados |
| `local_stratified_patch_v2` | gradiente local de densidade/composição | somente em fase posterior |
| `full_torus_gdml` | não previsto no ciclo inicial | não aprovado |

### 8.2 Forma do patch

O baseline será uma caixa centrada no vértice, com seis faces identificadas. Uma esfera poderá ser oferecida para testes de invariância rotacional. Toda saída de fronteira deve registrar:

- face/superfície;
- posição e normal locais;
- quatro-momento pré e pós-step;
- tempo local;
- material de origem;
- track, parent e processo.

### 8.3 Verificação geométrica

Cada build/teste deve executar:

- `CheckOverlaps` para volumes físicos;
- teste de ponto interno no vértice;
- raios analíticos até cada face;
- partículas em ±x, ±y, ±z;
- rotação conhecida da tétrada;
- tolerância de boundary sem duplicar nem perder crossings.

## 9. Lista física, cortes e controles

### 9.1 Baseline candidato

`FTFP_BERT` será o primeiro baseline de engenharia porque é uma lista de referência recomendada pelo Geant4 para aplicações de alta energia. Isso não constitui validação UHE do HADROS3.

Opções a comparar em fixtures:

- `FTFP_BERT` puro;
- `FTFP_BERT` com uma configuração EM explicitamente escolhida;
- lista com tratamento HP de nêutrons, somente se dados e custo forem justificados;
- modelos alternativos apenas com referência de domínio e teste thin-target.

O manifest deve guardar nome completo da lista, construtores registrados, versão e dump efetivo de parâmetros.

### 9.2 Production cuts não são tracking cuts

Os cortes de produção controlam a criação explícita de secundários abaixo de um range; eles não devem ser descritos como um corte que simplesmente mata tracks. O projeto registrará:

- cut por região e por espécie;
- energia equivalente calculada pelo material;
- `max_step` quando usado;
- qualquer limite de tempo/track como aproximação distinta;
- energia contabilizada localmente quando secundários não são produzidos.

### 9.3 Limites de segurança

Configurações explícitas:

- máximo de eventos por request;
- máximo de tracks por evento;
- máximo de steps por track/evento;
- máximo de tempo local;
- máximo de volume de saída;
- política para evento patológico;
- timeout externo do job.

Ultrapassar limite gera `resource_limit_exceeded`, nunca `ok`.

## 10. Scoring e balanço energético

### 10.1 Quantidades por evento

- energia total inicial das primárias;
- energia depositada por ionização/não ionizante quando disponível;
- energia total das partículas escapantes;
- energia de tracks terminados e sua razão;
- energia de partículas recusadas/não transportadas;
- variação de energia de repouso e termos nucleares necessários;
- residual bruto e residual explicado;
- número de steps, tracks, secundários e crossings;
- contagem e energia por processo, PDG e região.

### 10.2 Balanço

Não será usada a identidade ingênua `E_in = E_dep + E_escape` em eventos nucleares sem contabilizar massa de repouso, binding e tracks suspensos/terminados. O relatório deve decompor:

```text
E_in
  = E_escape
  + E_depositada
  + delta_E_resto_nuclear
  + E_terminada_contabilizada
  + E_nao_transportada
  + residual_nao_explicado
```

O campo que decide aceite é o residual **não explicado**. Componentes nunca podem ser descartados para forçar fechamento.

### 10.3 Ações Geant4

- `RunAction`: metadados globais e agregados thread-safe;
- `EventAction`: ledger e decisão de aceite por evento;
- `TrackingAction`: linhagem, início/fim e peso;
- `SteppingAction`: boundary crossings e processos;
- scorer/sensitive detector: deposição por volume ou voxel.

Arquivos grandes de steps ficam desligados por default. A saída de produção deve agregar voxels/processos e preservar somente registros necessários ao H3-W12.

## 11. Contrato de saída

```text
GEANT4/
  geant4_manifest.json
  geant4_environment_manifest.json
  geant4_jobs.jsonl
  geant4_summary.json
  geant4_summary.csv
  geant4_validation_report.json
  geant4_events_summary.jsonl
  geant4_escaped_particles.jsonl
  geant4_energy_deposition.jsonl
  geant4_process_counts.json
  geant4_unsupported_particles.jsonl
  geant4_energy_balance.png
  geant4_deposition_map.png
  geant4_process_composition.png
  geant4_escape_spectrum.png
  geant4_event_view.html
  jobs/<job_id>/
    effective_config.json
    stdout.log
    stderr.log
    raw_events.jsonl
```

Publicação deve usar `GEANT4/.staging-<run-id>` e rename atômico somente após validação. Falha preserva logs diagnósticos, mas não substitui um produto oficial anterior válido.

### 11.1 Fronteira para H3-W12

Cada registro escapante deve conter no mínimo:

```json
{
  "event_generation_event_id": "H3PWHG-000001:1",
  "pdg_id": 22,
  "position_local_mm": [0.0, 0.0, 10.0],
  "momentum_local_gev": [0.1, 0.2, 4.0],
  "energy_local_gev": 4.006,
  "time_local_ns": 0.03,
  "boundary_normal_local": [0.0, 0.0, 1.0],
  "track_id": 81,
  "parent_track_id": 7,
  "creator_process": "...",
  "generator_weights": {},
  "geant4_statistical_weight": 1.0,
  "local_tetrad_ref": "..."
}
```

H3-W12 aplicará a transformação local -> Kerr e a propagação ao observador. H3-W11 não deve pré-selecionar apenas fótons que apontam para a câmera.

## 12. Configuração proposta

```json
{
  "geant4": {
    "mode": "disabled",
    "backend": "geant4",
    "input_source": "event_generation_hepmc3",
    "geometry_model": "local_homogeneous_patch_v1",
    "patch_shape": "box",
    "patch_half_size_cm": [1.0, 1.0, 1.0],
    "density_source": "dis_vertex_local",
    "composition_model": "not_configured",
    "physics_list": "FTFP_BERT",
    "physics_domain_policy": "fail_stage",
    "production_cut_mm": 0.1,
    "max_step_mm": 1.0,
    "field_model": "none_local_inertial_patch",
    "optical_physics_enabled": false,
    "neutrino_policy": "pass_through",
    "random_seed": 59001,
    "seed_mode": "base_plus_stable_event_id",
    "threads": 1,
    "max_events": 2,
    "max_tracks_per_event": 100000,
    "max_steps_per_event": 1000000,
    "write_steps": false,
    "write_escaped_particles": true,
    "failure_policy": "fail_stage"
  }
}
```

`composition_model=not_configured` deve impedir transporte real. O preset padrão permanece `mode=disabled`.

### 12.1 Modos

| Modo | Efeito |
|---|---|
| `disabled` | não executa e não inventa resultados |
| `environment_check` | verifica binário, libs e datasets |
| `import_check` | lê HepMC3, unidades, PDGs, IDs e domínio; não transporta |
| `vacuum_smoke` | transporta até dois eventos/fixtures em vácuo |
| `material_smoke` | material validado, até dois eventos suportados |
| `real_free` | todos os eventos dentro de limites explícitos e domínio aprovado |

Um evento H3-W10 UHE real provavelmente falhará o guard inicialmente. Isso é o comportamento correto.

## 13. Integração CLI, Makefile e web

### 13.1 CLI

Adicionar a `hadros_web.py`:

```text
--run-geant4
--geant4-environment-check
--geant4-import-check
--geant4-vacuum-smoke
--geant4-material-smoke
```

CLI e web devem chamar a mesma função Python e produzir os mesmos manifests.

### 13.2 API

Endpoints propostos:

```text
POST /api/geant4/run
GET  /api/geant4/status
POST /api/geant4/cancel
```

Execução só começa após POST explícito. Abrir/recarregar a aba não pode executar nada. O processo deve ter PID/run ID próprio, lock por output root e status persistido.

### 13.3 Aba GEANT4

A aba deverá mostrar:

- estado: disabled, ready, running, complete, stale, failed ou unsupported-domain;
- versões Geant4/HepMC3 e datasets;
- entrada H3-W10 e hash;
- geometria, matéria, densidade e espessura de coluna;
- lista física, cuts, seed e threads;
- contagem supported/pass-through/unsupported;
- eventos, tracks, steps, energia depositada/escapante;
- maior residual energético não explicado;
- plots e visualizador;
- mensagem específica quando UHE exceder o domínio.

Não usar um banner "not running" para esconder resultados concluídos. Estado de processo e estado do último produto são campos diferentes.

### 13.4 Invalidação

Nova execução válida de H3-W10 invalida:

```text
GEANT4 -> PhotonTransport -> Spectra
```

Nova execução de H3-W11 invalida:

```text
PhotonTransport -> Spectra
```

Arquivos não devem ser apagados silenciosamente; devem ser marcados stale com causa e hashes divergentes.

## 14. Proveniência e reprodutibilidade

O manifest H3-W11 deve registrar:

- commit e dirty state do HADROS3;
- hash da configuração efetiva;
- hash de cada entrada H3-W10;
- versões e hashes do Geant4, HepMC3, compilador e executável;
- lista/versão de cada dataset;
- lista física e parâmetros efetivos;
- material, composição, densidade, geometria e cuts;
- seed base e seed efetivo por evento;
- número de threads e tipo de run manager;
- envelope de validade usado para cada espécie/processo;
- flags `geant4_invoked`, `photon_transport_invoked=false`, `spectra_invoked=false`;
- tempos de parede/CPU e contadores de recursos;
- estado completo, parcial, falho ou fora de domínio.

Para multithreading, seeds devem ser derivados do ID estável do evento, não da ordem em que uma worker recebeu o evento. O baseline de validação será single-thread; MT só entra após equivalência demonstrada.

## 15. Campanha de testes numéricos

Nenhum ponto da campanha de implementação receberá `COMPLLETO` apenas por compilar. Cada item exige o teste correspondente, artefato e tolerância aprovados.

### G4-T01 — Ambiente e datasets

**Teste:** inicializar a lista física e consultar todos os datasets exigidos.  
**Aceite:** versão fixada; zero dataset ausente; manifest contém paths, versões e hashes; processo retorna 0.

### G4-T02 — Round-trip de unidades e eixos

**Teste:** seis partículas em ±x, ±y, ±z, com quatro-momentos conhecidos.  
**Aceite:** direção preservada; erro relativo de energia/momento `<= 1e-12` antes do transporte; determinante da base positivo; nenhuma inversão norte/sul ou troca de eixo.

### G4-T03 — Cardinalidade e identidade HepMC3

**Teste:** comparar eventos e partículas finais entre HepMC3, sidecars e primárias Geant4.  
**Aceite:** bijeção de eventos; zero ID duplicado/ausente; PDG, carga, massa e quatro-momento correspondentes.

### G4-T04 — Transporte em vácuo

**Teste:** partículas estáveis atravessam caixa de vácuo.  
**Aceite:** energia depositada zero dentro da precisão; uma saída por primária não-decay; posição de crossing igual à solução reta com erro geométrico `<= 1e-9` relativo ao tamanho do patch; quatro-momento preservado.

### G4-T05 — Decaimento conhecido

**Teste:** partícula instável com decaimento Geant4 habilitado em domínio suportado.  
**Aceite:** vida média/amostragem compatível com referência no nível estatístico pré-definido; soma de branching ratios normalizada; energia residual não explicada dentro da tolerância.

### G4-T06 — Atenuação de fótons em slab

**Teste:** feixe monoenergético em material NIST e várias espessuras. Comparar fração de sobrevivência com `exp(-mu x)`.  
**Aceite:** resultado dentro de `3 sigma` binomial e diferença sistemática `<= 2%` após estatística suficiente; monotonicidade com coluna.

### G4-T07 — Poder de parada de carregadas

**Teste:** elétron/múon/próton monoenergético em slab fino, comparando `dE/dx` a referência Geant4/NIST aplicável.  
**Aceite:** média dentro de `2%` ou da incerteza da referência, o maior; convergência ao reduzir espessura e step.

### G4-T08 — Comprimento de interação hadrônica

**Teste:** prótons/píons dentro da faixa suportada em alvo fino.  
**Aceite:** probabilidade de interação compatível com a seção de choque/lista selecionada dentro de `3 sigma`; nenhum teste acima do envelope documentado.

### G4-T09 — Ledger energético EM

**Teste:** eventos somente EM em volume fechado e aberto.  
**Aceite inicial:** residual não explicado normalizado `<= 1e-6` por evento e quantil 99%; tolerância será apertada se o baseline demonstrar margem.

### G4-T10 — Ledger hadrônico/nuclear

**Teste:** eventos hadrônicos suportados com decomposição de repouso/binding.  
**Aceite inicial:** residual não explicado `<= 1e-4`, zero componente não finito e documentação de cada termo. Essa tolerância não pode ser relaxada sem relatório físico.

### G4-T11 — Convergência de cuts e step

**Teste:** repetir fixtures com `cut` e `max_step` reduzidos por fatores de 2.  
**Aceite:** energia total depositada, fuga e contagens físicas de interesse convergem dentro de `1%`; custo e tamanho são registrados.

### G4-T12 — Escala de densidade/coluna

**Teste:** mesma composição e comprimentos/densidades que preservam ou variam a coluna.  
**Aceite:** configurações de mesma coluna concordam dentro da incerteza no regime apropriado; interação cresce monotonicamente com coluna.

### G4-T13 — Fronteira H3-W12

**Teste:** crossing em cada face e canto, incluindo secundário.  
**Aceite:** exatamente um registro por crossing de saída; normal correta; on-shell; linhagem completa; reconstrução do evento sem perda de peso.

### G4-T14 — Guard de domínio UHE

**Teste:** píon/fóton/elétron/próton acima do limite configurado e PDG desconhecido.  
**Aceite:** produção falha antes de `BeamOn`; lista todas as violações; zero evento rotulado transportado; `quarantine_and_report` só funciona em modo diagnóstico.

### G4-T15 — Neutrinos pass-through

**Teste:** neutrinos em vácuo e matéria com política default.  
**Aceite:** nenhuma interação ou deposição; um registro escapante por primária; energia e ID preservados.

### G4-T16 — Determinismo

**Teste:** duas execuções single-thread com mesma seed e outra com seed diferente.  
**Aceite:** saídas canônicas idênticas byte a byte para mesma seed; hash diferente e observáveis estatisticamente compatíveis para seed diferente.

### G4-T17 — Equivalência multithread

**Teste:** 1 e N threads com seeds por ID estável.  
**Aceite:** mesma associação evento/seed e mesmos resultados canônicos quando suportado; no mínimo equivalência estatística documentada. MT fica desabilitado até passar.

### G4-T18 — Geometria e overlaps

**Teste:** `CheckOverlaps`, origem, raios analíticos e rotações.  
**Aceite:** zero overlap; origem sempre no material pretendido; erro de distância conforme G4-T04.

### G4-T19 — Pesos

**Teste:** eventos de peso positivo e negativo, mais pesos Observer Bridge distintos.  
**Aceite:** valores e sinais preservados sem multiplicação implícita; transporte analógico não altera pesos upstream.

### G4-T20 — Falhas, staging e stale

**Teste:** interromper job, corromper HepMC3, alterar hash H3-W10 e exceder limite de recursos.  
**Aceite:** nenhum produto parcial publicado como oficial; motivo correto; último resultado válido preservado e marcado stale quando aplicável.

### G4-T21 — Web/API

**Teste:** abrir e recarregar aba; disparar POST; consultar status; cancelar; reiniciar servidor.  
**Aceite:** nenhuma execução em GET/page load; um POST inicia no máximo um run; estado sobrevive ao reload; resultado concluído não é mostrado como "não rodou".

### G4-T22 — Integração H3-W10 -> H3-W11

**Teste:** fixture H3-W10 suportado atravessa importação, transporte, validação e publicação.  
**Aceite:** hashes encadeados, cardinalidade preservada, relatório `ok`, `geant4_invoked=true` e estágios posteriores ainda falsos.

## 16. Campanha de implementação ponto a ponto

O marcador `COMPLLETO` será escrito **na frente** do título somente depois de todos os critérios do ponto passarem. Neste plano inicial todos permanecem pendentes.

### G4-0 — Congelar contrato científico e numeração

- corrigir o nome `detector_transport_future` para transporte material local;
- atualizar o contrato H3-W11 e teoria sem afirmar implementação;
- decidir composição, tamanho/coluna do patch e política UHE;
- versionar schema de entrada/saída.

**Aceite:** revisão do contrato, schemas válidos e testes estáticos de numeração/campos.

### COMPLLETO — G4-1 — Bootstrap do Geant4

- instalar/fixar Geant4 11.4.2 e datasets no ambiente;
- criar detecção CMake e manifest;
- compilar/inicializar aplicação mínima headless.

**Aceite:** G4-T01 e build limpo.

### COMPLLETO — G4-2 — Importador HepMC3 e guard de domínio

- implementar reader, unidades, IDs, PDGs e seleção final;
- implementar tabela de domínio e políticas;
- emitir relatório sem executar transporte.

**Aceite:** G4-T02, T03 e T14.

### COMPLLETO — G4-3 — Geometria de vácuo e boundary writer

- implementar world/box, primárias, ações e crossing;
- preservar linhagem, pesos e tétrada;
- criar saída JSONL canônica.

**Aceite:** G4-T04, T13, T15, T18 e T19.

### G4-4 — Materiais e lista física

- implementar factory de materiais e composição versionada;
- registrar lista física, cuts e datasets;
- validar fixtures EM/hadrônicos dentro do domínio.

**Aceite:** G4-T05--T08.

### G4-5 — Scoring e ledger

- energia depositada, fuga, processos, trajetórias e termos nucleares;
- agregação por evento/material/voxel;
- limites de recursos e saída opcional de steps.

**Aceite:** G4-T09, T10 e limites testados em T20.

### G4-6 — Patch astrofísico local

- mapear densidade do vértice e composição aprovada;
- converter escala física e tétrada;
- testar coluna, cuts, step e tamanho do patch.

**Aceite:** G4-T11 e T12, mais relatório de convergência.

### G4-7 — Orquestrador Python e produtos

- jobs, staging, hashes, agregação, plots, viewer e relatório;
- locks, cancelamento e recuperação de falha;
- integração em paths, pipeline e proveniência.

**Aceite:** G4-T16, T20 e T22 single-thread.

### G4-8 — CLI, Makefile e aba web

- comandos e endpoints explícitos;
- renderer específico e estados separados de processo/produto;
- invalidação downstream e links de outputs.

**Aceite:** G4-T21 e regressão da aba Event Generation.

### G4-9 — Multithreading e desempenho

- run manager MT, seeds por evento e writers thread-safe;
- benchmark de memória, CPU e volume de saída;
- preservar single-thread como referência.

**Aceite:** G4-T17, sem regressão numérica acima das tolerâncias.

### G4-10 — Decisão e validação UHE

- auditar espécie por espécie nas energias reais do HADROS3;
- selecionar/acoplar modelo externo ou parametrização quando necessário;
- comparar com dados, publicações ou geradores de referência;
- executar campanha por década de energia;
- documentar envelope final de produção.

**Aceite:** nenhum evento real fora de domínio; relatório científico aprovado. Sem isso, H3-W11 fica `supported_energy_only`, não `production_complete`.

### G4-11 — Release e atualização da teoria

- executar suíte completa em ambiente limpo;
- arquivar manifests, versões e artefatos numéricos;
- atualizar `HADROS3_Physics_Theory` e campanha de física;
- marcar cada G4-N elegível com `COMPLLETO`.

**Aceite:** G4-T01--T22, `make validate`, documentação e reprodutibilidade independente.

## 17. Ordem de execução recomendada

```text
G4-0 contrato
  -> G4-1 bootstrap
  -> G4-2 importador/domain guard
  -> G4-3 vácuo e fronteira
  -> G4-4 materiais/lista física
  -> G4-5 scoring/ledger
  -> G4-6 patch astrofísico
  -> G4-7 orquestração
  -> G4-8 web
  -> G4-9 MT
  -> G4-10 UHE
  -> G4-11 release
```

G4-8 não deve preceder a validação do backend; a interface não será usada para mascarar placeholder como resultado. G4-10 pode começar em paralelo conceitual desde G4-0, mas sua aprovação depende dos dados e comparações científicas.

## 18. Critério de conclusão do H3-W11

H3-W11 só poderá ser chamado `implemented` quando:

1. Geant4 e datasets estiverem fixados e reproduzíveis;
2. a entrada HepMC3 tiver bijeção e unidades/referencial validados;
3. matéria e geometria tiverem significado astrofísico documentado;
4. lista física e domínio forem declarados por espécie e energia;
5. balanços, atenuação, dE/dx, interação e convergência passarem;
6. pesos e proveniência forem preservados;
7. a saída H3-W12 estiver completa e on-shell;
8. falhas/parciais/stale não puderem parecer sucesso;
9. web não iniciar jobs implicitamente;
10. a faixa real de energias HADROS3 estiver coberta ou explicitamente bloqueada;
11. todos os pontos aceitos estiverem marcados `COMPLLETO` com evidência.

Se apenas fixtures abaixo de 100 TeV passarem, o estado correto será:

```text
H3-W11 implemented_for_supported_energy_validation_only
H3-W11 UHE production blocked_by_physics_domain
```

## 24. Visualização ligada macro/local — implementação de 2026-07-14

COMPLLETO — O backend agora registra passos dentro de `Patch`, incluindo posições pré/pós-passo, track e parent IDs, PDG, energias, deposição, processo que definiu o passo, processo criador, secundárias, volumes e flags de interação/fronteira.

COMPLLETO — `geant4_macro_sites_3d.html` mostra BH, toro analítico, funis polares e os pontos exatos aceitos pelo DIS. Cada ponto conserva `interaction_id`, coordenadas `r, theta, phi`, posição cartesiana em `r_g`, densidade, evento GEANT4 e link para o volume local.

COMPLLETO — clicar em um ponto abre `geant4_event_view.html?event=N` em nova janela. O segundo visualizador desenha o cubo local em milímetros, tracks primários, tracks secundários, vértices/processos físicos e a contagem de escapes.

COMPLLETO — a interface declara explicitamente a separação de escalas. O cubo local é uma aproximação homogênea no referencial tangente e não é desenhado no tamanho do sistema BH+toro. Para `M=3 M_sun`, o half-size atual de `10 mm` corresponde a aproximadamente `2.2573e-6 r_g`.

COMPLLETO — arquivos grandes são controlados por `max_recorded_steps` (default `50000`). O viewer usa no máximo `20000` passos, preservando até metade do orçamento para interações e amostrando o restante. Tanto o truncamento do registro quanto a amostragem do display são mostrados ao usuário.

COMPLLETO — a política de densidade do material ficou auditável: `HADROS3_H_HE` recebe a densidade local do DIS; materiais NIST, inclusive `G4_Galactic`, usam a densidade interna do Geant4 e são identificados como tal na interface e no manifest.

Aceite numérico da execução corrente:

| Quantidade | Resultado |
|---|---:|
| ponto macro | `H3DIS-000005` |
| posição BL-like | `r=13.6017147207 r_g`, `theta=0.7592829642`, `phi=6.6966678433` |
| posição cartesiana | `(8.5743568690, 3.7622383012, 9.8657290957) r_g` |
| densidade aplicada ao H/He | `134718.10774393746 g/cm3` |
| passos Geant4 totais | `1825449` |
| passos registrados | `50000` (`steps_truncated=true`) |
| passos colocados no viewer | `20000` |
| partículas escapantes | `3296` |
| energia depositada | `9506.780451764698 GeV` |

Testes de aceite após a implementação:

- `tests/test_geant4_transport.py`: `20 passed`;
- `tests/test_hadros_web.py`: `16 passed`;
- renderização real dos dois HTMLs em Chrome headless: passou;
- determinismo byte a byte inclui agora `geant4_steps_raw.jsonl`;
- teste geométrico verifica a transformação esférica/cartesiana e o link macro → volume local.

Esta seção substitui, para a amostra corrente de `10^4 GeV`, o diagnóstico histórico de bloqueio UHE registrado nas seções 22–23. O guard continua testado e continua bloqueando entradas que excedam `100000 GeV`.

## 19. Riscos e mitigação

| Risco | Efeito | Mitigação/aceite |
|---|---|---|
| Eventos muito acima do domínio da lista física | resultado numericamente executado, fisicamente inválido | guard bloqueante e G4-10 |
| Composição do toro ausente | material arbitrário | G4-0 exige prescrição antes de material real |
| Patch local pequeno/grande demais | sub ou superestima interação/fuga | convergência em coluna/tamanho |
| Misturar Geant4 local com Kerr global | orientação e trajetória erradas | tétrada local e fronteira H3-W12 separada |
| Optical physics explosiva | custo e produto sem interpretação | desligada por default |
| Cuts tratados como tracking cuts | deposição e multiplicidade mal interpretadas | registro de cuts e convergência |
| Naive energy ledger em reações nucleares | falso erro ou falsa conservação | decomposição completa G4-T10 |
| Pesos NLO/observador misturados | taxas enviesadas | namespaces e G4-T19 |
| MT muda associação seed/evento | irreprodutibilidade | seed por ID e single-thread baseline |
| Arquivos de steps enormes | disco/memória saturados | agregação, limites e opt-in |
| Aba dispara job ao abrir | execução inesperada | POST explícito e G4-T21 |
| Resultado antigo parece atual | cadeia inconsistente | hashes e stale downstream |

## 20. Decisões científicas ainda necessárias

O início de G4-1 e G4-2 independe destas decisões; transporte astrofísico real não:

1. Qual composição nuclear representa o toro em cada cenário?
2. Qual é a escala física/coluna do patch por vértice?
3. Como tratar hádrons, léptons e fótons acima dos envelopes Geant4?
4. Existe campo magnético local relevante no recorte inicial?
5. Quais partículas escapantes H3-W12 aceita além de fótons?
6. Qual precisão física é exigida por década de energia e espécie?

Defaults conservadores propostos:

- composição: `not_configured`, bloqueante;
- campo: nenhum;
- óptica: desligada;
- threads: 1;
- política fora de domínio: falhar;
- output de steps: desligado;
- geometria científica inicial: patch homogêneo local;
- nenhum transporte em Kerr dentro do Geant4.

## 21. Referências primárias

- [Geant4 11.4.2 — página oficial de download e datasets](https://geant4.web.cern.ch/download/)
- [Release notes do Geant4 11.4](https://geant4.web.cern.ch/download/release-notes/notes-v11.4.0.html)
- [Geant4 Installation Guide 11.4](https://geant4.web.cern.ch/documentation/dev/ig_html/InstallationGuide/index.html)
- [Book for Application Developers 11.4](https://geant4.web.cern.ch/documentation/dev/bfad_pdf/BookForApplicationDevelopers.pdf)
- [Geometria: volumes sólidos, lógicos e físicos](https://geant4.web.cern.ch/documentation/pipelines/master/bfad_html/ForApplicationDevelopers/Detector/Geometry/geomIntro.html)
- [Materiais no Geant4](https://geant4.web.cern.ch/documentation/pipelines/master/bfad_html/ForApplicationDevelopers/Detector/material.html)
- [Ações obrigatórias e listas físicas de referência](https://geant4.web.cern.ch/documentation/pipelines/master/bfad_html/ForApplicationDevelopers/UserActions/mandatoryActions.html)
- [Guia oficial da lista FTFP_BERT e seus intervalos de energia](https://geant4.web.cern.ch/documentation/pipelines/master/plg_html/PhysicsListGuide/reference_PL/FTFP_BERT.html)
- [Production thresholds versus cuts](https://geant4.web.cern.ch/documentation/dev/bfad_html/ForApplicationDevelopers/TrackingAndPhysics/thresholdVScut.html)
- [Scoring por comandos e primitive scorers](https://geant4.web.cern.ch/documentation/dev/bfad_html/ForApplicationDevelopers/Detector/commandScore.html)
- [Design de multithreading e random seeds](https://geant4.web.cern.ch/documentation/pipelines/master/bftd_html/ForToolkitDeveloper/OOAnalysisDesign/Multithreading/mt.html)
- [Exemplos básicos oficiais do Geant4](https://geant4.web.cern.ch/documentation/pipelines/master/bfad_html/ForApplicationDevelopers/Examples/BasicCodes.html)

## 22. Evidência da implementação executada em 2026-07-13

Ambiente instalado e fixado:

- Geant4 `11.4.2`, HepMC3 `3.3.1`, CMake e Ninja no ambiente `dis`;
- todos os datasets oficiais exigidos foram encontrados e registrados no manifest;
- backend C++ headless compilado com `FTFP_BERT` e orquestrador Python integrado ao hadros-web;
- configuração default permanece `disabled`; abrir/recarregar a aba ou consultar por `GET` não inicia transporte;
- a execução exige `POST /api/geant4` com `action=run_geant4` ou comando explícito do Makefile.

Resultados numéricos reproduzidos pela suíte `tests/test_geant4_transport.py`:

| Ensaio | Resultado observado | Aceite automatizado |
|---|---:|---|
| Vácuo, seis direções cartesianas | 6/6 crossings nas faces corretas; deposição `< 1e-20 GeV` | passou |
| Atenuação de fótons de 1 GeV em Pb | sobrevivência `0.782`, `0.500`, `0.282` em 2, 5 e 10 mm; coeficientes `0.123`, `0.139`, `0.127 mm^-1` | passou o baseline de consistência e monotonicidade |
| Múons em água | deposição média `1.7328e-4` e `3.5205e-4 GeV` em 1 e 2 mm; razão `2.0317` | passou a escala linear `1.9--2.1` |
| Prótons de 10 GeV em Pb | sobrevivência `0.954`, `0.864`, `0.744` em 10, 30 e 50 mm | passou consistência de comprimento de interação |
| Decaimento de píon carregado em repouso | produtos `mu+ + nu_mu`, linhagem Geant4 e soma `0.13957039 GeV` | passou com residual relativo `< 1e-12` |
| Determinismo serial | três produtos canônicos idênticos byte a byte para seed igual | passou |
| Suíte Geant4 focal | `17 passed` | passou |
| Regressão completa HADROS3 | `110 passed, 4 skipped` em `167.97 s` | passou; skips são testes opcionais condicionados a backends/fixtures externos |

Auditoria do produto H3-W10 real em
`output/HADROS3_hadros_web_preview/EventGeneration/event_generation_events.hepmc3`:

- 1 evento e 57 primárias finais foram importados sem perda de cardinalidade;
- densidade do vértice H3-W7 resolvida para `134718.10774393746 g/cm3`;
- 22 das 57 primárias excedem o teto validado de `100000 GeV`;
- energia máxima encontrada: `339735762.44114596 GeV`;
- o guard retornou `unsupported_domain`, `geant4_invoked=false` e zero evento transportado, antes de `BeamOn`.

Os marcadores `COMPLLETO` foram mantidos apenas nos pontos cujo aceite atual foi realmente satisfeito. G4-4 continua pendente porque os baselines implementados ainda não atingem todas as comparações externas e tolerâncias de T05--T08; G4-7 continua pendente pelo cancelamento/lock persistente exigido no aceite; G4-5/G4-6/G4-8 ainda requerem os testes completos de ledger nuclear, convergência e cancelamento web; G4-9 é deliberadamente serial; G4-10 requer uma decisão científica UHE.

## 23. Resultado atual do plano

H3-W11 está implementado, integrado e numericamente testado **para o domínio de energia suportado**. Ele não é declarado `production_complete` para os eventos UHE reais: o evento disponível alcança aproximadamente `3.40e8 GeV`, enquanto o envelope conservador configurado para `FTFP_BERT` termina em `1.00e5 GeV`. O comportamento correto e comprovado nessa condição é bloquear antes do transporte e publicar o diagnóstico, não extrapolar silenciosamente a física.

Estado científico publicado:

```text
H3-W11 implemented_for_supported_energy_validation_only
H3-W11 UHE production blocked_by_physics_domain
```
