# HADROS3: neutrinos UHE e transporte em matéria densa

Esta pasta reúne a bibliografia usada para avaliar os estágios H3-W10
(PYTHIA 8/HepMC3) e H3-W11 (GEANT4) nas densidades do toro. Os PDFs foram
obtidos de repositórios abertos dos autores, arXiv ou CERN Document Server.

## Produtos

- `HADROS3_UHE_Dense_Matter_Review.pdf`: revisão crítica e plano de validação;
- `HADROS3_UHE_Dense_Matter_Review.tex`: fonte LaTeX do relatório;
- `references.bib`: metadados e links persistentes;
- `papers/`: cópias locais dos artigos;
- `SHA256SUMS`: hashes das cópias efetivamente analisadas.

## Organização da literatura

| Tema | Referências principais | Uso no HADROS3 |
|---|---|---|
| Geração de eventos | Bierlich et al. (PYTHIA 8.3) | shower, hadronização, decaimentos, matching e auditoria do evento isolado |
| DIS de neutrinos em TeV--UHE | Weigel et al.; Xie et al.; Le e Mäntysaari | referências para seção de choque e distribuição de inelasticidade |
| Dados do LHC | FASER 2024 e 2025 | medida direta de interações de neutrinos na faixa TeV |
| GEANT4 e calorimetria | Allison et al.; CALICE 2013, 2016 e 2019 | validação laboratorial de chuveiros e modelos FTFP_BERT |
| Fontes astrofísicas densas | Murase e Ioka; Senno, Murase e Mészáros | resfriamento e reinteração de secundários em jatos ocultos |
| Matéria densa | Chamel e Haensel; Potekhin et al.; Klein; Fu et al. | degenerescência, correlações, stopping coletivo e supressões ambientais |

## Nota de validade

Nenhum dos trabalhos experimentais encontrados reproduz matéria H/He com
densidade de `10^9--10^10 g cm^-3`. A bibliografia permite validar partes da
cadeia e construir testes de extrapolação, mas não transforma o material
homogêneo padrão do GEANT4 em um modelo comprovado de plasma degenerado.
