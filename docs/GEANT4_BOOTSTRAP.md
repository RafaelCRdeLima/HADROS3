# H3-W11 GEANT4 bootstrap

## Versões fixadas

- Geant4 11.4.2;
- HepMC3 3.3.1;
- lista física inicial `FTFP_BERT`;
- build C++17 por CMake/Ninja;
- execução serial até a equivalência multithread ser validada.

## Instalação

```bash
python scripts/geant4/bootstrap_geant4.py install
python scripts/geant4/bootstrap_geant4.py inspect
make geant4-build
make geant4-validate
```

O bootstrap instala também os datasets distribuídos pelo pacote conda-forge. `inspect` exige todos os diretórios registrados em `geant4_environment_manifest.json`.

## Execução

```bash
make geant4-environment-check
make geant4-import-check
make geant4-vacuum-smoke
make geant4-material-smoke
```

Nenhum comando é executado ao abrir a aba GEANT4. A interface exige o botão **Run GEANT4**, que envia `action=run_geant4` por POST.

## Limite UHE

O baseline usa teto conservador de `100000 GeV`. Uma partícula acima dele produz `unsupported_domain`, código 3 no backend e nenhum `BeamOn`. Não aumente `validated_maximum_energy_gev` para fazer um evento passar: o schema oficial proíbe valores acima desse teto enquanto G4-10 não tiver um modelo UHE validado.

## Referencial e matéria

As primárias são interpretadas em `local_matter_tetrad`, GeV e mm. A geometria é um patch cartesiano local; Kerr permanece fora do Geant4. Em execução oficial, `density_source=dis_vertex_local` recupera a densidade do registro H3-W7 que originou o evento H3-W10. `configured_fixture` existe apenas para testes controlados.

## Diagnóstico

- `geant4_import_report.json`: cardinalidade, unidades, PDGs e violações de domínio;
- `geant4_environment_manifest.json`: versões, binário e datasets;
- `geant4_validation_report.json`: critérios numéricos;
- `geant4_stderr.log`: falha do toolkit;
- `geant4_unsupported_particles.jsonl`: partículas recusadas.
