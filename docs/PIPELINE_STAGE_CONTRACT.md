# HADROS3 Pipeline Stage Contract

This document defines the stage contract for new HADROS3 pipeline phases.
It is documentation only: it does not require refactoring existing stages.

The contract pattern is:

```text
inputs -> run() -> outputs -> diagnostics -> provenance
```

It applies to future stages such as:

```text
H3-W9 Event Generation
H3-W10 GEANT4
H3-W11 Photon Transport
H3-W12 Spectra
```

The existing implemented stages remain as they are:

```text
H3-W5 UHE Source
H3-W6 Forward Geodesics
H3-W7 DIS Interaction Sampler
H3-W8 Observer Bridge
```

They should not be refactored merely to match this document.

## Required Contract

Every new HADROS3 stage should define and maintain these elements.

1. Official inputs

   The stage must consume only official outputs from earlier stages, plus the
   run configuration and run metadata. It must not depend on temporary files,
   dashboard-only state, local developer paths, or undocumented side effects.

2. Dedicated output directory

   The stage must write only to its own directory under:

   ```text
   output/<run-name>/<StageName>/
   ```

   It must not modify output folders owned by earlier stages.

3. Clear data products

   Data products should have stable names and formats. Use JSONL for event-like
   records, JSON for structured summaries and reports, and CSV for simple tabular
   summaries intended for quick inspection.

4. Automatic diagnostics

   The stage target must generate its diagnostics automatically as part of the
   normal stage run. A separate dashboard button may exist for convenience, but
   it must not be the only way to create required diagnostics.

5. Provenance

   The stage must record enough provenance to audit what ran, what inputs were
   consumed, what outputs were created, and which implementation path was used.

6. Expensive-stage flags

   The stage must explicitly record whether expensive or future event-generation
   systems were invoked. Use false unless the stage actually called them.

7. Tests

   The stage must include tests for input discovery, output generation, reports,
   provenance fields, and dashboard integration. Tests should prefer temporary
   run directories and should not depend on persistent `output/` contents.

8. `hadros_web` integration

   The stage must appear in the dashboard with clear run controls, status,
   summaries, diagnostics, and Outputs tab entries. Dashboard state must not
   contaminate static presets.

   H3-W8b is presented visually as `Gravitational Image Analysis`. Its
   compatibility output directory remains `ObserverImageBranches/`, and its
   reports must make clear that `branch_score` is a proxy ranking score rather
   than true magnification, radiative-transfer intensity, or detector response.

9. Makefile target

   The stage must have an explicit Makefile target. Running that target should
   perform the complete stage action, including required diagnostics and reports.

10. Non-modification rule

    A stage may read official earlier outputs, but it must not rewrite, patch,
    clean, or append to output folders owned by previous stages.

## Stage Interface

Each stage should be describable as:

```text
official inputs
-> run(config, input_paths, output_dir)
-> official outputs
-> diagnostics
-> provenance/report
```

The run function may be implemented in Python, C++, CUDA, or a combination of
backends. The report must record which backend path was used.

## Required Metadata

Each new stage report or provenance block should record:

```text
stage_name
stage_version_or_phase
backend_name
backend_language
backend_binary_or_module
run_name
config_path
input_paths
output_dir
output_products
diagnostics_generated
random_seed
random_seed_source
uses_physics_proxy
physics_proxy_model
physics_proxy_risk
known_limitations
```

If a field is not applicable, record a clear false or null value rather than
omitting it silently.

## Expensive And Future Pipeline Flags

Every new stage should record these flags in its summary/report/provenance:

```text
powheg_invoked
pythia_invoked
geant4_invoked
photon_transport_invoked
expensive_event_generation_invoked
```

When relevant, also record:

```text
event_generation_invoked
external_backend_invoked
external_backend_name
external_backend_command
external_backend_return_code
```

These flags are part of the safety boundary between diagnostic/proxy stages and
full event-generation or transport stages.

## Physical Proxies And Risks

Any proxy model must be explicit. Reports should state:

```text
uses_physics_proxy = true
physics_proxy_model = <model name>
physics_proxy_risk = true
```

Examples include geometric visibility proxies, escape-probability proxies,
redshift proxies, or camera-plane projection proxies. Proxy outputs must not be
presented as full POWHEG, PYTHIA, GEANT4, or photon-transport results.

## Theory Documentation Requirement

The implemented physics reference is:

```text
docs/Theory/HADROS3_Physics_Theory.tex
docs/Theory/HADROS3_Physics_Theory.pdf
```

Any stage that implements new physics must update both the LaTeX source and the
regenerated PDF. The update must document:

```text
physical inputs consumed by the stage
implemented equations
approximations and proxies
statistical weights
outputs passed to the next stage
physical risks and limitations
```

A stage must not be considered complete if its implemented physics is absent
from the theory document. If the work is infrastructure, UI, refactoring, or a
visual diagnostic that adds no new physics, the commit or review note should
state explicitly that the theory PDF did not need an update.

If a stage changes equations or statistical weights already documented in the
theory PDF, the PDF must be updated in the same commit or in a separate commit
immediately before the physics change. Planned or proposed physics must not be
documented as implemented until it is part of the official pipeline.

Every new physical implementation must also update:

```text
VERSION.json
Theory Version
Compatible HADROS3 commit
implemented-stage table
provenance theory fields
```

The theory document uses `major.minor` versioning:

```text
major
```

is incremented when the implemented physics changes incompatibly, for example a
breaking change to DIS optical depth, geodesic propagation, physical weights, or
medium semantics.

```text
minor
```

is incremented when a new physical stage, physical backend, or formulation is
implemented in a backward-compatible way.

Examples:

```text
1.0 -> 1.1  Observer Bridge implemented
1.1 -> 1.2  POWHEG real run implemented
1.2 -> 2.0  incompatible DIS or geodesic physics change
```

Changes limited to UI, dashboard controls, plots, infrastructure, build system,
tests, formatting, or visual diagnostics do not need to increment the Theory
Version unless they alter equations, approximations, weights, physical models,
statistics, proxies, or physical backends. Such commits should say explicitly
that the Theory Version did not need to change.

## Scientific Release Metadata

The repository root must contain `VERSION.json` with:

```text
software_version
physics_version
pipeline_version
theory_version
theory_document
```

Every provenance record must include a `scientific_release` block carrying those
values plus the current `git_commit`.

Before merging a new physical stage, evaluate whether it increments:

```text
physics_version
theory_version
pipeline_version
```

Changes that are purely visual, dashboard-only, infrastructure, build-system, or
test-related may increment only `software_version`. Changes to equations,
approximations, physical models, statistical weights, proxies, physical
statistics, or physical backends must update the relevant scientific release
versions and the Theory PDF.

Scientific release versions must be changed through:

```bash
make release-software
make release-physics
make release-pipeline PIPELINE=H3-W9b
```

These targets call `scripts/release/update_version.py`, which is the official
way to edit `VERSION.json`. It updates release date and git commit metadata and
keeps the Theory LaTeX metadata synchronized before rebuilding the PDF.

Use:

```text
make release-software
```

for infrastructure, UI, dashboard, build-system, test, documentation, or visual
diagnostic changes that do not change implemented physics.

Use:

```text
make release-physics
```

for changes to equations, approximations, physical models, statistical weights,
proxies, physical statistics, or physical backends.

Use:

```text
make release-pipeline PIPELINE=...
```

when a new physical stage becomes the most advanced implemented stage. After
any release update, run:

```bash
make validate
```

## Presets And Runtime State

Static presets, including:

```text
presets/hadros_web/default_config.json
```

must remain static. Stage execution, dashboard actions, and validation commands
must not write runtime status, local paths, preview state, or generated values
into static presets.

Runtime configuration snapshots belong under run-owned metadata locations such
as:

```text
output/<run-name>/RunMetadata/
output/<run-name>/RunMetadata/configs/
```

or another documented run-owned directory.

## Dashboard Contract

For each new stage, `hadros_web` should expose:

```text
stage action button
input availability/status
requested parameters
summary metrics
diagnostic figures
report/provenance links
Outputs tab entries
```

Dashboard text should distinguish proxy diagnostics from physical simulation
results when a proxy model is used.

## Makefile Contract

Each new stage should have a Makefile target with a stable name. The target
should:

```text
load the current run config
consume only official inputs
write only to the stage output directory
generate required diagnostics
write summary/report/provenance
leave earlier outputs unchanged
```

Running the target repeatedly should be deterministic when the same config,
inputs, and seeds are used.

## Testing Contract

Tests for a new stage should verify:

```text
official inputs are discovered
missing inputs fail clearly
outputs are generated in the stage directory
previous stage directories are not modified
diagnostics are generated automatically
provenance fields are present
expensive-stage flags are false unless invoked
dashboard lists stage outputs
Makefile target behavior is covered directly or through the orchestrator
```

Where possible, tests should use temporary directories and small fixtures.

## Future Application

This contract should guide the implementation of:

```text
H3-W9 Event Generation
H3-W10 GEANT4
H3-W11 Photon Transport
H3-W12 Spectra
```

It is intended to prevent accidental coupling between stages, silent mutation of
previous outputs, ambiguous proxy semantics, and accidental invocation of costly
simulation backends.
