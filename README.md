# HADROS3

HADROS3 currently starts with the `hadros-web` control shell: a visual and
configuration-first base for the Kerr black hole, observer camera, analytic
torus, polar cone/funnel, UHE source region, placeholders, outputs, and
provenance.

This stage does not run POWHEG, PYTHIA, GEANT4, forward neutrino geodesics,
optical-depth DIS, or an active observer bridge.

## Commands

Open the HADROS3 web control dashboard:

```bash
make hadros-web
```

This serves the control UI at:

```text
http://127.0.0.1:8877
```

The run name is chosen in the top bar of the dashboard; outputs are written
automatically under `output/<run-name>/`. The former Run tab was removed.
For now, render commands are coordinated from `make`; run buttons will be added
later once the run flow is defined.

The Camera tab supports these preview modes:

```text
analytic_geometry_only
kerr_like_cuda
full_kerr
```

The CUDA modes reuse the original HADROS `hadros_geodesic_preview_cuda`
backend when it is available. If it is not available at runtime, HADROS3 writes
a clear fallback summary and a geometry-only camera preview.

HADROS3 vendors the visual assets used by the original config-web under
`assets/`: `assets/logo/Hadros_logo.png` and the sky panorama in
`assets/sky/eso0932a.{jpg,ppm}`. The dashboard serves these through `/assets/`.

Render the preview/configuration products directly and exit:

```bash
make render-hadros-web
```

Render only the camera preview:

```bash
make render-camera-preview
```

Generate the H3-W5 UHE source samples through the central `hadros_web`
orchestrator:

```bash
make sample-uhe-source
```

The same action is available in the dashboard under the **UHE Source** tab as
`Generate UHE Source Samples`. It writes the source products under
`output/<run-name>/UHEsource/`:

```text
uhe_neutrino_source_samples.jsonl
uhe_neutrino_source_summary.csv
uhe_neutrino_source_preview.png
```

Open the original HADROS interactive camera preview window:

```bash
make launch-camera-preview
```

This delegates to the original HADROS target:

```text
make -C ../HADROS geodesic_preview
```

with HADROS3 camera, black-hole, torus, and funnel parameters passed through as
`PREVIEW_*`, `ASPIN`, and `CAM_*` variables. Runtime preview files are written
to a space-safe `/tmp/hadros3_camera_preview_*` folder because the original
HADROS Makefile does not quote output paths; the HADROS3 launcher log remains in
`output/<run-name>/interactive_camera_preview/camera_preview_interactive.log`.

Controls are inherited from HADROS: drag/arrows orbit the camera, scroll or
`+/-` changes distance, `[]` changes FOV, `A/D` changes spin, `R` renders, `S`
saves when supported, and `Q`/`Esc` closes. Camera JSON files are saved by the
original preview in `../HADROS/configs/cameras/`.

For HADROS3 the preview launcher explicitly passes `PREVIEW_DISK_R_IN_RG`,
`PREVIEW_DISK_R_OUT_RG`, and `PREVIEW_DISK_THICKNESS_RG` from the configured
analytic torus so the inherited HADROS preview disk does not extend out to the
camera by default.

`make serve-hadros-web` is kept as an alias for `make hadros-web`.

Run syntax checks:

```bash
make check
```

Remove generated preview products and Python caches:

```bash
make clean
```

The default preview target writes products to the folder derived from the run
name, for example:

```text
output/HADROS3_hadros_web_preview/
```
