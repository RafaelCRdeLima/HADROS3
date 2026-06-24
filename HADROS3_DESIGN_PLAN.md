# HADROS3 Design Plan

## 1. Purpose

HADROS3 is a proposed next-generation redesign of the HADROS-CASCADE UHE
neutrino-to-photon-observer pipeline.

The central physical change is:

```text
Current HADROS-CASCADE sampler:
  observer-linked Kerr geodesics
  -> choose interaction points along those geodesics
  -> generate DIS/cascade products

HADROS3 target sampler:
  physical UHE neutrino source
  -> forward neutrino propagation
  -> optical-depth interaction sampling
  -> observer-bridge weighting
  -> expensive event generation only for accepted interactions
```

HADROS3 should preserve the validated strengths of HADROS-CASCADE while
removing the main physical weakness of the current workflow: interaction points
are currently observer-guided rather than generated from an explicit UHE
neutrino source population.

The first implementation priority is therefore not POWHEG, PYTHIA, GEANT4, or
even the new optical-depth sampler. The first product of HADROS3 is a separate
`hadros-web` configuration and control surface: a visual, configurable, and
provenance-aware system equivalent in spirit to the original HADROS
`config_web.py`/`config_web_final.py`, but owned by HADROS3. This first layer
must make the Kerr black hole, camera, analytic torus, polar cone/funnel, UHE
emission regions, observer geometry, and schematic system preview explicit
before any expensive event is generated.

This document is design only. It does not implement code.

Out of scope for this design document:

```text
new executable code
new POWHEG/PYTHIA integration code
new GEANT4 code
full radiative transfer
detector response
Compton opacity
pair cascade
real Uribe velocity field
production physical gamma-gamma opacity
```

## 2. Core Physical Picture

The intended physical chain is:

```text
black hole + torus/NDAF/funnel
-> UHE neutrino source near polar funnel or funnel wall
-> forward null geodesic propagation of neutrinos
-> neutrino interaction with baryonic matter by DIS
-> POWHEG/PYTHIA hard event and shower/hadronization
-> GEANT4 local cascade
-> photon propagation to distant observer
-> validated ZAMO redshift
-> observer spectra and science products
```

In HADROS3, the observer should not create the interaction point. The observer
should enter as a selection and weighting stage after a physical source and
interaction model have produced candidate events.

## 3. Guiding Design Principles

1. Separate source physics from observer selection.
2. Generate UHE neutrino candidates from an explicit source model.
3. Propagate neutrinos forward in Kerr spacetime.
4. Sample interactions using optical depth, not arbitrary geometric proximity.
5. Use observer-connected geodesics only as an importance-sampling bridge.
6. Run expensive POWHEG/PYTHIA/GEANT4 only after cheap physics filters.
7. Preserve event weights and sampling PDFs explicitly.
8. Keep ideal, toy, diagnostic, and physical products separate.
9. Keep configuration centralized; no hidden defaults.
10. Every physically incomplete approximation must be visible in provenance.

## 4. Proposed Architecture

### 4.0 HADROS-Web Control Layer

Before the source Monte Carlo layer, HADROS3 should create a dedicated
`hadros-web` layer.

Purpose:

```text
central configuration
visual geometry preview
schematic physical-system drawing
Kerr camera preview/control
analytic torus and polar-cone setup
provenance/config export
safe handoff to later source, geodesic, sampler, and bridge stages
```

Main file:

```text
hadros_web.py
```

The file should be separate from the original HADROS web interface and should
command the HADROS3 configuration. It should be visually organized like the
current `config_web_final.py`, but should expose HADROS3 concepts directly.

The first visible/configurable HADROS3 system should include:

```text
Kerr black hole
observer/camera
field of view
observer distance
inclination
image and preview resolution
analytic torus
polar cone/funnel
UHE emission region
neutrino energy controls
schematic geometry of black hole + torus + cone + observer
observer bridge controls, initially diagnostic
output/provenance controls
```

Reusable HADROS-CASCADE components to import or port where validated:

```text
Kerr infrastructure
main camera
preview camera
FOV controls
observer distance controls
inclination controls
schematic angle/size/cone/torus visualization
layout and organization patterns from config_web_final.py
centralized provenance/config writing
```

The control layer should make clear which parameters control:

```text
black_hole
spin
mass
camera_distance
inclination
field_of_view
resolution
analytic_torus
polar_cone
uhe_neutrino_source
neutrino_energy
interaction_sampler
observer_bridge
outputs
```

Success criterion for this first layer:

```text
HADROS3 opens through hadros_web.py and can configure and draw the complete
physical system before any expensive POWHEG/PYTHIA/GEANT4 event is generated.
```

### 4.1 Source Monte Carlo Layer

After the `hadros-web` control layer exists, the first physics-sampling layer
defines and samples UHE neutrino emission.

Candidate source models:

```text
axial_point
polar_cone
funnel_wall
axial_cap
disk_corona
custom_source_table
```

Minimal first implementation target:

```text
polar_cone
```

because it is physically interpretable, simple, and provides direct control of
opening angle and emission direction.

Each sampled neutrino source record should contain:

```text
source_sample_id
event_id
source_model
x_emit_t
x_emit_r
x_emit_theta
x_emit_phi
p_nu_emit_t
p_nu_emit_r
p_nu_emit_theta
p_nu_emit_phi
E_nu_emit_gev
E_nu_inf_gev
source_physical_pdf
source_sampling_pdf
source_weight
source_status
```

The key statistical weight is:

```text
source_weight = source_physical_pdf / source_sampling_pdf
```

for biased or importance-sampled emission.

### 4.2 Forward Neutrino Geodesic Layer

After source sampling, neutrino trajectories are propagated forward as null
geodesics.

Stop conditions:

```text
horizon_crossing
outer_escape_radius
torus_exit
max_affine_length
max_steps
invalid_invariant
```

Diagnostics:

```text
max_null_norm_abs
relative_E_killing_error
relative_Lz_error
geodesic_status
```

The forward neutrino layer should not produce observer photons. It only
produces neutrino paths and candidate matter crossings.

### 4.3 DIS Optical-Depth Interaction Sampler

For each forward neutrino path, integrate the neutrino-nucleon optical depth:

```text
d tau_nuN = n_baryon(x) * sigma_nuN(E_nu_local) * dl

tau_nuN_path = integral d tau_nuN

P_int = 1 - exp(-tau_nuN_path)
```

where:

```text
n_baryon = rho / m_baryon
E_nu_local = -p_mu u_medium^mu
```

If a neutrino interacts, sample the interaction location from:

```text
p(s | interaction) proportional to d tau_nuN / ds
```

This is physically superior to uniformly selecting points or selecting only
observer-linked ray samples.

Required DIS fields:

```text
tau_nuN_total
interaction_probability
interaction_accepted
interaction_sample_u
interaction_affine_parameter
interaction_r_rg
interaction_theta_rad
interaction_phi_rad
E_nu_local_gev_at_interaction
sigma_GBW_cm2
sigma_IIM_cm2
dis_model
interaction_weight
```

### 4.4 Observer Bridge Layer

The observer bridge is the computational efficiency layer.

It uses a precomputed bundle of observer-connected Kerr geodesics to estimate
whether a candidate interaction is likely to contribute to the observer camera.

It must not replace the source model. It is only an importance filter.

Candidate bridge tests:

```text
distance_to_observer_geodesic_bundle
momentum_alignment_with_observer_bundle
solid_angle_proxy
lensing_bundle_density_proxy
field_of_view_compatibility
```

The bridge produces:

```text
observer_connection_weight
observer_bridge_status
observer_pixel_candidates
bridge_sampling_pdf
bridge_physical_pdf
bridge_weight
```

The final event weight should include:

```text
final_event_weight =
  source_weight
  * interaction_weight
  * observer_bridge_weight
```

The exact normalization must be reviewed before claims of absolute flux.

### 4.5 Expensive Event Gate

Only after a candidate interaction passes cheap source, optical-depth, and
observer-bridge checks should HADROS3 run expensive event generation:

```text
POWHEG DIS
-> PYTHIA8
-> GEANT4 local cascade
```

The gate should support:

```text
generate_powheg
reuse_validated_lhe
reuse_run_local_event_record
diagnostic_no_expensive_event
```

but every mode must be declared in provenance.

### 4.6 Photon Observer Layer

HADROS3 should reuse the validated HADROS-CASCADE photon observer stack:

```text
photon escape classifier
observer sphere hits
camera projection
validated ZAMO redshift
photon spectra
photon diagnostics
optional separated opacity products
```

This reuse is intentional. The major redesign is upstream of photon production,
not in the photon observer itself.

## 5. Proposed Pipeline

Target HADROS3 production pipeline:

```text
1. Launch hadros_web.py
2. Configure black hole, torus, polar cone, source, observer, and outputs
3. Draw visual/schematic geometry preview
4. Precompute observer geodesic bundle
5. Sample UHE neutrino source
6. Propagate neutrinos forward
7. Integrate DIS optical depth
8. Sample accepted interaction points
9. Apply observer-bridge weighting/prefilter
10. Run POWHEG/PYTHIA/GEANT4 for accepted interactions
11. Propagate photons to observer
12. Validate ZAMO redshift
13. Build spectra and science products
14. Write provenance and trust-boundary summaries
```

Minimal early pipeline:

```text
1. hadros_web.py
2. Kerr black hole configuration
3. observer camera, FOV, distance, inclination, and preview resolution
4. analytic_torus configuration and drawing
5. polar_cone/funnel configuration and drawing
6. UHE emission-region configuration and schematic drawing
7. geometry preview and centralized provenance/config export
8. no POWHEG/PYTHIA/GEANT4 yet
9. no optical-depth DIS sampler yet
```

## 6. Configuration Design

HADROS3 should keep a single source of truth, analogous to
`config_web_final.py`, but separated from the current production interface until
stable. The first implementation of this source of truth should be
`hadros_web.py`.

Proposed config groups:

```text
run
hadros_web
black_hole
observer_camera
geometry_preview
torus_medium
polar_cone
uhe_neutrino_source
forward_neutrino_geodesics
dis_interaction_sampler
observer_bridge
event_generation
photon_observer
outputs
```

### 6.0 HADROS-Web Parameters

```text
hadros_web_host
hadros_web_port
hadros_web_mode
hadros_web_preview_mode
hadros_web_provenance_path
hadros_web_config_export_path
geometry_preview_enabled
geometry_preview_style
geometry_preview_show_black_hole
geometry_preview_show_torus
geometry_preview_show_polar_cone
geometry_preview_show_uhe_region
geometry_preview_show_observer
geometry_preview_show_camera_frustum
geometry_preview_show_angle_annotations
```

### 6.1 Black Hole And Camera Parameters

```text
black_hole_metric
black_hole_mass_msun
black_hole_spin_a
black_hole_rg_cm
observer_distance_rg
observer_inclination_deg
observer_azimuth_deg
camera_fov_deg
camera_static_resolution
camera_preview_resolution
camera_preview_quality
camera_geodesic_backend
```

### 6.2 Analytic Torus And Polar Cone Parameters

```text
torus_model
torus_r_inner_rg
torus_r_outer_rg
torus_r_peak_rg
torus_half_opening_angle_deg
torus_density_model
torus_density_norm_g_cm3
torus_show_in_preview
polar_cone_opening_angle_deg
polar_cone_theta_min_deg
polar_cone_theta_max_deg
polar_cone_r_min_rg
polar_cone_r_max_rg
polar_cone_show_in_preview
polar_cone_draw_mode
```

### 6.3 UHE Source Parameters

```text
uhe_source_model
uhe_source_energy_model
uhe_source_energy_min_gev
uhe_source_energy_max_gev
uhe_source_energy_mono_gev
uhe_source_spectral_index
uhe_source_opening_angle_deg
uhe_source_theta_min_deg
uhe_source_theta_max_deg
uhe_source_r_min_rg
uhe_source_r_max_rg
uhe_source_phi_mode
uhe_source_n_samples
uhe_source_sampling_mode
uhe_source_seed
```

Allowed first source models:

```text
polar_cone
axial_point
```

Future source models:

```text
funnel_wall
axial_cap
source_table
```

### 6.4 Forward Geodesic Parameters

```text
uhe_neutrino_max_steps
uhe_neutrino_step_rg
uhe_neutrino_max_radius_rg
uhe_neutrino_horizon_tolerance_rg
uhe_neutrino_invariant_tolerance
uhe_neutrino_fail_on_invalid
```

### 6.5 DIS Sampler Parameters

```text
uhe_dis_model
uhe_dis_sigma_gbw_path
uhe_dis_sigma_iim_path
uhe_dis_medium_model
uhe_dis_interaction_sampling_mode
uhe_dis_min_tau_for_candidate
uhe_dis_max_candidates
uhe_dis_fail_on_oob_medium
```

Recommended first interaction sampling mode:

```text
optical_depth_inverse_cdf
```

### 6.6 Observer Bridge Parameters

```text
observer_bridge_mode
observer_bridge_distance_tolerance_rg
observer_bridge_angle_tolerance_deg
observer_bridge_min_connection_weight
observer_bridge_max_pixel_candidates
observer_bridge_weight_model
observer_bridge_fail_on_unweighted_acceptance
```

Allowed first bridge modes:

```text
disabled
diagnostic_distance_to_bundle
importance_prefilter
```

Recommended first mode:

```text
diagnostic_distance_to_bundle
```

to audit bias before using the bridge to reject events.

## 7. Data Products

### 7.1 Source Samples

```text
uhe_neutrino_source_samples.jsonl
```

Required fields:

```text
event_id
source_sample_id
source_model
x_emit_t
x_emit_r
x_emit_theta
x_emit_phi
p_emit_t
p_emit_r
p_emit_theta
p_emit_phi
E_nu_emit_gev
source_physical_pdf
source_sampling_pdf
source_weight
source_status
```

### 7.2 Forward Neutrino Paths

```text
uhe_neutrino_forward_path_segments.jsonl
```

This should be compressed or on-the-fly by default. Raw JSONL may be used for
small audits only.

Required fields:

```text
event_id
source_sample_id
segment_index
r_start_rg
theta_start_rad
phi_start_rad
r_end_rg
theta_end_rad
phi_end_rad
r_mid_rg
theta_mid_rad
phi_mid_rad
p_t_mid
p_r_mid
p_theta_mid
p_phi_mid
dl_segment_rg
E_nu_local_gev_mid
rho_g_cm3_mid
n_baryon_cm3_mid
sigma_nuN_cm2_mid
d_tau_nuN
geodesic_status
```

### 7.3 Interaction Candidates

```text
uhe_neutrino_interaction_candidates.jsonl
```

Required fields:

```text
event_id
source_sample_id
tau_nuN_total
interaction_probability
interaction_candidate_status
candidate_r_rg
candidate_theta_rad
candidate_phi_rad
candidate_E_nu_local_gev
candidate_sigma_nuN_cm2
candidate_medium_status
```

### 7.4 Accepted Interactions

```text
uhe_neutrino_interaction_accepted.jsonl
```

Required fields:

```text
event_id
source_sample_id
interaction_id
interaction_accepted
interaction_r_rg
interaction_theta_rad
interaction_phi_rad
interaction_E_nu_local_gev
source_weight
interaction_weight
observer_bridge_weight
final_event_weight
bridge_status
expensive_event_generation_status
```

### 7.5 Summaries And Provenance

```text
uhe_neutrino_source_summary.csv
uhe_neutrino_interaction_summary.csv
uhe_observer_bridge_summary.csv
hadros3_pipeline_provenance.json
```

## 8. Statistical Weights

HADROS3 must make all statistical weights explicit.

Minimum weight decomposition:

```text
w_final =
  w_source
  * w_interaction
  * w_observer_bridge
  * w_event_generation
```

where:

```text
w_source = physical source PDF / sampling source PDF
w_interaction = interaction probability or conditional sampling correction
w_observer_bridge = correction for observer-importance sampling or bridge prefilter
w_event_generation = correction from reusing event records or LHE samples
```

No absolute luminosity or flux claim should be made until the weight
normalization is audited.

## 9. Validation Plan

### 9.1 Source Validation

Tests:

```text
sampled positions lie inside configured source volume
sampled directions lie inside configured cone
source PDFs are finite and positive
weights are finite
energy distribution matches configured spectrum
```

### 9.2 Geodesic Validation

Tests:

```text
null norm within tolerance
Killing energy conservation
Lz conservation
stop conditions explicit
no hidden truncation
```

### 9.3 DIS Optical-Depth Validation

Tests:

```text
rho >= 0
n_baryon >= 0
sigma_nuN >= 0
d_tau >= 0
tau_total >= 0
P_int = 1 - exp(-tau_total)
constant-density analytic case reproduces tau = n sigma L
```

### 9.4 Observer Bridge Validation

Tests:

```text
bridge weights finite and non-negative
events close to observer bundle get larger weights than far events
disabled bridge preserves all accepted interactions
diagnostic bridge does not reject events
prefilter bridge records every rejection
```

### 9.5 End-To-End Validation

Tests:

```text
toy source + toy medium produces reproducible weighted event counts
accepted events can feed existing POWHEG/PYTHIA/GEANT4 handoff
photon observer products preserve event weights
spectra include weighted and unweighted summaries
```

## 10. Performance Strategy

The design avoids brute force by using cheap filters before expensive event
generation.

Cost hierarchy:

```text
cheap:
  source sampling
  forward geodesic propagation
  medium lookup
  optical-depth integration

moderate:
  observer bridge distance/angle query

expensive:
  POWHEG
  PYTHIA
  GEANT4
  photon propagation for many secondaries
```

HADROS3 should only run expensive stages after:

```text
source sample valid
geodesic valid
interaction accepted
observer bridge weight above diagnostic threshold or explicitly retained
```

Acceleration options:

```text
compressed path segments
on-the-fly optical-depth accumulation
spatial index for observer geodesic bundle
batch medium lookup
GPU geodesic preview/propagation where validated
event-cache reuse with provenance
```

## 11. Relationship To Current HADROS-CASCADE

HADROS3 should reuse:

```text
Kerr geodesic infrastructure where validated
GBW/IIM sigma tables
POWHEG/PYTHIA handoff
GEANT4 local cascade handoff
photon escape classifier
photon observer sphere hits
camera projection
validated ZAMO redshift
photon spectra
diagnostic and toy opacity infrastructure
```

HADROS3 should replace or redesign:

```text
observer-guided interaction point sampling
implicit neutrino source assumptions
unweighted observer-conditioned event selection
```

## 12. Baby-Step Implementation Roadmap

### Phase H3-W0: Create Separate HADROS3 Module

Implement:

```text
separate HADROS3 folder/module boundary
HADROS3-owned configuration namespace
clear import boundary from original HADROS-CASCADE
initial provenance/config paths
```

Goal:

```text
HADROS3 can evolve without modifying the current production HADROS web stack.
```

### Phase H3-W1: Port Kerr, Camera, And Preview Foundations

Import or port from validated HADROS-CASCADE code:

```text
Kerr infrastructure
main camera
preview camera
FOV controls
observer distance controls
inclination controls
preview resolution controls
schematic angle/size/cone/torus visualization patterns
centralized provenance/config writing patterns
```

No new neutrino sampler yet.

### Phase H3-W2: Create hadros_web.py

Implement:

```text
HADROS3 web entry point
central configuration controls
black-hole/camera controls
torus controls
polar-cone controls
UHE-source placeholder controls
sampler/bridge/output controls, initially disabled or diagnostic
config export
provenance export
```

The interface should be visually clear, legible, and functional in the same
spirit as `config_web_final.py`, but should be a separate HADROS3 surface.

### Phase H3-W3: Implement Black Hole, Analytic Torus, And Polar Cone

Implement:

```text
Kerr black hole preview
analytic_torus geometry
polar cone/funnel geometry
UHE emission-region drawing
observer/camera frustum drawing
parameter validation for geometric consistency
```

No POWHEG/PYTHIA/GEANT4.

### Phase H3-W4: Generate Schematic System Figure

Implement:

```text
schematic black hole + torus + cone + observer figure
angle annotations
size/radius annotations
camera/FOV annotation
source-region annotation
exportable preview image
provenance record for all figure parameters
```

First-stage success criterion:

```text
hadros_web.py opens and can configure and draw the complete physical setup
before any expensive event is generated.
```

### Phase H3-W5: Start polar_cone Source Sampler

Begin the physics sampler only after the HADROS3 visual/configurable shell
exists.

Implement:

```text
polar_cone source
monoenergetic neutrinos
source weights
source summary outputs
```

### Phase H3-W6: Start Forward Geodesics

Implement:

```text
forward Kerr neutrino geodesics
path diagnostics
invariant validation
```

### Phase H3-W7: Start Optical-Depth DIS Sampler

Implement:

```text
analytic_torus medium lookup
GBW/IIM sigma lookup
tau_nuN integration
optical-depth interaction sampling
```

No expensive event generation yet.

### Phase H3-S0: Documentation And Audit

Status:

```text
design only, superseded as the first implementation step by H3-W0 through H3-W4
```

Deliverables:

```text
HADROS3_DESIGN_PLAN.md
current sampler audit
source/observer bridge math note
```

### Phase H3-S1: Source Sampler Prototype

This phase corresponds to H3-W5 in the revised order. It should not start until
`hadros_web.py` can configure and draw the black hole, camera, analytic torus,
polar cone, and UHE emission region.

Implement:

```text
polar_cone source
monoenergetic neutrinos
source weights
no geodesic propagation yet
```

Outputs:

```text
uhe_neutrino_source_samples.jsonl
uhe_neutrino_source_summary.csv
```

### Phase H3-S2: Forward Geodesic Prototype

This phase corresponds to H3-W6 in the revised order.

Implement:

```text
forward null geodesic propagation
compressed path segments
invariant validation
```

No DIS yet.

### Phase H3-S3: Optical-Depth DIS Sampler

This phase corresponds to H3-W7 in the revised order.

Implement:

```text
analytic_torus medium
GBW/IIM sigma lookup
tau_nuN integration
interaction sampling
```

No POWHEG/PYTHIA/GEANT4 yet.

### Phase H3-S4: Observer Bridge Diagnostic

Implement:

```text
precomputed observer geodesic bundle
distance-to-bundle diagnostics
connection weight diagnostics
no rejection by default
```

### Phase H3-S5: Observer Bridge Prefilter

Implement:

```text
importance prefilter
explicit rejection records
bridge weights
weighted summaries
```

### Phase H3-S6: Expensive Event Gate

Connect accepted interactions to:

```text
POWHEG
PYTHIA
GEANT4
existing photon observer pipeline
```

### Phase H3-S7: Physical Medium Upgrade

Add:

```text
Uribe radial scalar if audited
Uribe reconstructed disk only with explicit closure
real u_fluid only if data/model exists
```

### Phase H3-S8: Physical Opacity Upgrade

Only after the source and bridge are validated:

```text
physical gamma-gamma opacity
Compton/Klein-Nishina
pair cascade
radiation-field contracts
```

## 13. Physics Risk Register

| Risk | Consequence | Mitigation |
|---|---|---|
| Source PDF not normalized | Wrong flux normalization | Store source PDFs and weights explicitly. |
| Bridge rejects real observable events | Biased spectra | Start bridge as diagnostic-only. |
| Observer bundle too coarse | Missed lensing paths | Use convergence tests with increasing camera resolution. |
| Medium lacks real u_fluid | Wrong local neutrino energy | Mark ZAMO fallback as physics risk. |
| Reused LHE events misweighted | Wrong event statistics | Track event-generation weights and provenance. |
| Optical depth sampled on truncated path | Nonphysical interaction probability | Fail by default on truncation. |
| Expensive event gate too aggressive | Artificially hard selection | Keep rejected-event summary and thresholds. |

## 14. Recommended First Technical Decision

The first concrete HADROS3 implementation should not touch POWHEG, PYTHIA, or
GEANT4. It also should not start with the optical-depth DIS sampler.

Recommended first implementation:

```text
Phase H3-W0 through H3-W4:
  create a separate HADROS3 module/folder
  port/import validated Kerr + camera + preview foundations
  create hadros_web.py
  configure black hole, spin, mass, camera distance, inclination, FOV, resolution
  configure and draw analytic torus
  configure and draw polar cone/funnel
  configure and draw UHE emission region
  export centralized config and provenance
  no POWHEG/PYTHIA/GEANT4 yet
  no optical-depth sampler yet
```

Recommended second implementation:

```text
Phase H3-W5 / H3-S1:
  polar_cone source sampler
  monoenergetic neutrinos
  explicit source weights
  no forward geodesics yet
```

Recommended third implementation:

```text
Phase H3-W6 / H3-S2:
  forward Kerr neutrino geodesics
  compressed path output
  invariant validation
```

Recommended fourth implementation:

```text
Phase H3-W7 / H3-S3:
  optical-depth DIS sampler on analytic_torus
```

This order first builds the correct HADROS3 shell: visual configuration, Kerr
geometry, camera preview, analytic structures, and provenance. Only then should
HADROS3 move into the new physical source sampler, forward geodesics, and
optical-depth DIS interaction sampling.

## 15. Readiness Statement

```text
Ready to create design folder?
  Yes.

Ready to implement code?
  Yes, first for H3-W0 through H3-W4: the hadros-web shell, geometry preview,
  Kerr/camera import, analytic torus, polar cone, and provenance/config export.

Ready to replace HADROS-CASCADE current sampler?
  No. Needs source sampler, forward geodesics, optical-depth validation,
  and observer bridge diagnostics.

Ready to run POWHEG/PYTHIA/GEANT4 through HADROS3?
  No. Expensive event generation should be gated after accepted
  interactions and validated bridge weights.

Most physical path without exploding compute:
  hadros_web.py visual/configuration shell
  + Kerr black hole/camera preview
  + analytic torus and polar cone geometry
  + provenance/config export
  then:
  forward source Monte Carlo
  + optical-depth interaction sampling
  + observer bridge diagnostics/prefilter
  + expensive event generation only after acceptance.
```
