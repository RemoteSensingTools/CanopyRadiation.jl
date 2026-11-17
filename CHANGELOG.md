# Changelog

All notable changes to `CanopyRadiation.jl` are documented in this file.

## Unreleased

Internal refactor of the whole package (stages 1 and 2). No exported name was removed
or renamed, no exported argument order or return field changed, and every output of
the frozen v0.1.0 numerical records (`test/frozen_record`: core, remaining public
API, CanopyOptics-dependent tables) is bit-identical on the tuple, vector and batch
paths. The version in `Project.toml` is not bumped.

### Added
- Package extension `CanopyRadiationForwardDiffExt` (weak dependency ForwardDiff
  0.10 or 1): defines the scalar hooks for `ForwardDiff.Dual`, so AD behaves
  identically on both ForwardDiff majors.

### Changed
- One precision policy (`src/precision.jl`) for every function that takes scalars,
  records or arrays of them: mixed `Float32`/`Float64` inputs and integer, rational,
  irrational or boolean scalars throw an `ArgumentError` with a static message
  (previously a `MethodError`). This includes `unit_sin_from_cos`,
  `hemispheric_bin`, `convert` between records of different precision, integer
  element types of workspaces, batch arrays, leaf emission and photon arrays, and the
  in-place hemisphere drivers when the output element type differs from the promoted
  input type. A `Float64` mixed with a `Float64`-valued dual number still promotes to
  the dual. The spectral basis functions and `build_leaf_emission_matrices` keep
  their `AbstractFloat` signatures.
- Number types with derivative information take part through five public, not
  exported, scalar hooks: `CanopyRadiation.primal_type`, `primal`, `all_finite`,
  `exact_zero` and `scalar_equal`. They replace the reflection helpers of the
  deleted `scalars.jl` (`primal_float_type`, `primal_scalar`, `is_continuous_scalar`,
  `compatible_scalar`, `promote_compatible_scalar`, `require_continuous_scalar`,
  `finite_scalar`, `exact_zero_scalar`); without hooks for a dual type, its numbers
  are rejected.
- Every record (two-stream methods, rates and operators, volume emissions, emitting
  layers, population quadrature and layouts, emission angular laws, emission
  boundary and atmospheres, leaf emission models, spectral maps, layer hemispheric
  sources) shares one definition of `==`, `isequal`, `hash` and broadcasting
  (`src/records.jl`), including records that hold host or device arrays: `-0.0 ==
  0.0`, `NaN != NaN`, `isequal` and `hash` consistent, array axes part of the value,
  and derivative components compared on both ForwardDiff majors (previously `==`
  followed the ForwardDiff version). Hash values differ from v0.1.0.
  `QuadratureEmission`, `PopulationQuadrature`, the leaf emission models, the
  emission boundary and atmosphere, the spectral maps and the layer hemispheric
  sources now broadcast as scalars. The exported abstract types are subtypes of the
  internal record supertype. `FullEmissionSupport` (a CanopyOptics subtype) keeps
  Base equality.
- Dispatch replaces type switches: method parameters in `layer_operator` and
  `interior_flux`; photosystem pairs of `DualLeafEmissionMatrix` (validation,
  escape, bin checks, bin axis, out-of-window ledger); the forms of
  `out_of_window_fraction` and the `StrictStokes` tail check; the CPU fast paths of
  `photon_bins!`, `photon_density!`, `photon_energy!`,
  `hemispheric_source_moments!` and `expand_hemispheric_source!` are methods on the
  backend type (the three photon conversions share one kernel).
- The passive and the emitting column each have one adding sweep
  (`src/column_sweeps.jl`), shared by `solve_column!` / `solve_emitting_column!` and
  the batch kernels (`src/column_batch.jl`) through column views; the tuple
  recursions stay.
- `adapt_radiation_storage` has one field-mapping method for every array-holding
  object instead of eleven (same objects, same results, `QuadratureEmission` and
  `LayerHemisphericSource` still rebuilt without value checks).
- Source files by process: `face_and_geometry.jl` became `leaf_face_absorption.jl`,
  `leaf_population.jl`, `emission_angular_law.jl` and `atmosphere_coupling.jl`;
  `hemispheric_sources.jl` became `hemispheric_moments.jl`, `sif_assembly.jl` and
  `device_storage.jl`; `buffer_aliasing.jl` holds the output aliasing checks;
  `scalars.jl` is deleted (including the unused `neumaier_add`).

### Known, unchanged
These inconsistencies are kept on purpose (each changes behaviour and needs a
decision):
- `batch_hemispheric_moments!` and `batch_expand_hemispheric_sources!` validate less
  than their scalar counterparts (no cosine range, finiteness or sign checks).
- `validate_leaf_emission(::DualLeafEmissionMatrix)` defaults to `atol = 1e-6`, the
  single-matrix method to `sqrt(eps)`.
- `diffuse_response` and the native thin slab use the small-q series for dual
  numbers but the closed form for floats, so a dual's value can differ from the
  plain result in the last bits.
- The batch matrices store the layers of a column contiguously; on GPUs the column
  per thread access is not coalesced.

## [0.1.0] - 2026-11-12

### Added
- Passive two-stream canopy-column solvers (`NativeTwoStream`, `AnalyticTwoStream`), `CanopyRates`, `foursail_canopy_rates`, `LayerOperator`, and pre-allocated `ColumnFluxWorkspace` (`solve_column`, `solve_column!`).
- Unified longwave thermal and fluorescence cavity solvers (`NoEmission`, `UniformEmission`, `QuadratureEmission`, `EmittingLayer`, `EmittingColumnWorkspace`, `solve_emitting_column`, `solve_emitting_column!`, `solve_emission`) with exact net and gross column conservation.
- Anatomical leaf-face illumination (`source_projection`, `beam_face_absorption`, `isotropic_face_absorption`, `regular_plate_passive`), sun/shade `PopulationQuadrature`, `LambertianFaces` / `MixedLambertianFaces` solid-angle quadrature, and `AngularEmissionBoundary` / `couple_emission`.
- Single- and dual-photosystem (`PSII + PSI`) conditional leaf spectral escape matrices (`LeafEmissionMatrix`, `DualLeafEmissionMatrix`, `FullEmissionSupport`, `build_leaf_emission_matrices`) with explicit out-of-window Stokes tail tracking.
- Spectral grid pullbacks and anchor bases (`IdentitySpectralMap`, `LinearSpectralMap`, `prepare_emission_basis`, `prepare_anchor_basis`).
- Hemispheric source projection (`LayerHemisphericSource`, `DualLayerHemisphericSource`, `evaluate_hemispheric_sif_bin`, `evaluate_hemispheric_sif_spectrum`), `KernelAbstractions` batch kernels, and `CanopyRadiationCUDAExt` (`adapt_radiation_storage`).
- Continuous-scalar `ForwardDiff.Dual` automatic differentiation support across conservative (`α → 0`), resonant (`k → m`), `LAI = 0`, and zenith/nadir (`|μ| = 1`) limits, plus `Documenter.jl` API reference and CI workflows.
