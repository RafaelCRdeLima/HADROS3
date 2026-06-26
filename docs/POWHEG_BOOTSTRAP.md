# H3-W9 POWHEG DIS Bootstrap

This document describes the first H3-W9 bootstrap step for a self-contained
POWHEG DIS setup inside HADROS3.

This is not yet the full Observer Bridge to POWHEG pipeline. It only verifies
that HADROS3 can fetch or copy POWHEG-BOX-RES, build the DIS `pwhg_main`
executable locally, and run a minimal smoke test that produces a real LHE file.

## Strategy

HADROS3 uses a fetch/build strategy instead of vendoring the full
POWHEG-BOX-RES source tree.

Reasoning:

```text
reproducibility: pinned upstream commits are recorded
repository size: the full POWHEG tree is large and should not be committed
licensing: external source remains external and auditable
GitHub compatibility: generated source/build artifacts stay ignored
```

Pinned sources:

```text
POWHEG-BOX-RES: https://gitlab.com/POWHEG-BOX/RES/POWHEG-BOX-RES.git
POWHEG-BOX-RES commit: 1a31cef9bc594e7f59ac94485a107512030dd1b1
DIS process: https://gitlab.com/POWHEG-BOX/RES/User-Processes/DIS.git
DIS commit: 29f394adf9e958c307812fd9e2ce61d368461e96
```

If a local audited POWHEG source exists at:

```text
../HADROS-CASCADE/external_cache/POWHEG-BOX-RES/
```

`make powheg-fetch` copies it into HADROS3. Otherwise it clones the pinned
upstream source.

## Commands

Fetch or copy the POWHEG source:

```bash
make powheg-fetch
```

Build the local DIS executable:

```bash
make powheg-build
```

Run a minimal smoke test:

```bash
make powheg-smoke
```

## Paths

Fetched source:

```text
external/powheg/POWHEG-BOX-RES/
```

Build output copied back into HADROS3:

```text
external/powheg/build/DIS/pwhg_main
external/powheg/build/powheg_build.log
external/powheg/build/powheg_build_summary.json
```

Smoke-test output:

```text
tmp/powheg_smoke/powheg.input
tmp/powheg_smoke/powheg.log
tmp/powheg_smoke/pwgevents.lhe
tmp/powheg_smoke/powheg_smoke_summary.json
```

These paths are ignored by git.

## Dependencies

The bootstrap requires:

```text
make
git, if cloning instead of copying a local audited source
Fortran compiler, usually gfortran
LHAPDF and lhapdf-config
NNPDF31_nlo_as_0118 PDF data, LHAPDF ID 303400
```

The current helper detects:

```text
LHAPDF_CONFIG
lhapdf-config on PATH
~/micromamba/envs/dis/bin/lhapdf-config
```

If `gfortran` is not on PATH, it tries the local `hadros-cascade` micromamba
environment for the build toolchain. This is a build-time convenience only; the
runtime smoke test calls the `pwhg_main` executable copied into HADROS3.

## Path With Spaces

The upstream POWHEG build is sensitive to paths with spaces. The HADROS3 helper
therefore stages the source under:

```text
/tmp/hadros3_powheg_build/
```

for compilation, then copies the resulting executable back to:

```text
external/powheg/build/DIS/pwhg_main
```

The smoke test calls only that local HADROS3 executable.

## Smoke Physics

The smoke card is intentionally tiny:

```text
numevts = 2
ih1 = 12
ih2 = 1
ebeam1 = 1e9 GeV
ebeam2 = 0.938272 GeV
channel_type = 3
vtype = 2
```

It checks that POWHEG DIS can produce an LHE file with:

```text
<LesHouchesEvents
</LesHouchesEvents>
```

and at least one `<event>` block.

## Scope And Risks

This bootstrap does not connect POWHEG to:

```text
ObserverBridge/
PYTHIA
GEANT4
Photon Transport
Spectra
```

It is a build and smoke-test layer only. The smoke LHE is not a production
physics sample.

Known risks:

```text
Fortran compiler may be absent
LHAPDF may be absent
PDF data may be absent
upstream POWHEG build may fail in paths with spaces
smoke integration settings are intentionally low statistics
```

## Cleaning Artifacts

Remove fetched sources, build outputs, and smoke products with:

```bash
rm -rf external/powheg/POWHEG-BOX-RES external/powheg/build tmp/powheg_smoke
```

Do not commit fetched POWHEG source, object files, `pwhg_main`, logs, or LHE
files.
