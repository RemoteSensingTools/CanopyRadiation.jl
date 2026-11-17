# Changelog

All notable changes to `CanopyRadiation.jl` are documented in this file.

## Unreleased

Internal refactor of the numeric core (first stage). No exported name, argument
order or return field changed, and every output of the frozen v0.1.0 numerical
record (`test/frozen_record`) is bit-identical on the tuple, vector and batch paths.
The version in `Project.toml` is not bumped.

### Changed
- Precision policy in one place (`src/precision.jl`) for the two-stream and emission
  core (`two_stream.jl`, `emission_transport.jl`) and the column batch kernels:
  mixed `Float32`/`Float64` inputs and integer, rational, irrational or boolean
  scalars now throw an `ArgumentError` with a static message (previously a
  `MethodError`). This includes `convert` between records of different precision
  and the integer element types of workspaces and batch arrays. A `Float64` mixed
  with a `Float64`-valued dual number still promotes to the dual. Functions in the
  other source files keep their previous behaviour for now.
- Number types with derivative information take part through five public, not
  exported, scalar hooks: `CanopyRadiation.primal_type`, `primal`, `all_finite`,
  `exact_zero` and `scalar_equal`. They replace the reflection helpers
  (`primal_float_type`, `primal_scalar`, …) in the converted files; without hooks for
  a dual type, its numbers are rejected by the converted functions.
- `NativeTwoStream`, `AnalyticTwoStream`, `CanopyRates`, `LayerOperator`,
  `NoEmission`, `UniformEmission`, `QuadratureEmission` and `EmittingLayer` share one
  definition of `==`, `isequal`, `hash` and broadcasting (`src/records.jl`): values
  compare equal with `-0.0 == 0.0` and `NaN != NaN`, `isequal` and `hash` are
  consistent, and derivative components are compared too, identically on
  ForwardDiff 0.10 and 1.x (previously `==` followed the ForwardDiff version). Hash
  values differ from v0.1.0. `QuadratureEmission` now broadcasts as a scalar.
  `AbstractTwoStreamMethod` and `AbstractVolumeEmission` are subtypes of the internal
  record supertype.
- `layer_operator` and `interior_flux` dispatch on the method for its parameters
  (no `isa` switch); `layer_operator(::AbstractTwoStreamMethod, …)` is one method.
- The passive and the emitting column each have one adding sweep
  (`src/column_sweeps.jl`), shared by `solve_column!` / `solve_emitting_column!` and
  the batch kernels through column views; the tuple recursions stay. The batch
  kernels and drivers moved from `hemispheric_sources.jl` to `src/column_batch.jl`.

## [0.1.0] - 2026-11-12

### Added
- Passive two-stream canopy-column solvers (`NativeTwoStream`, `AnalyticTwoStream`), `CanopyRates`, `foursail_canopy_rates`, `LayerOperator`, and pre-allocated `ColumnFluxWorkspace` (`solve_column`, `solve_column!`).
- Unified longwave thermal and fluorescence cavity solvers (`NoEmission`, `UniformEmission`, `QuadratureEmission`, `EmittingLayer`, `EmittingColumnWorkspace`, `solve_emitting_column`, `solve_emitting_column!`, `solve_emission`) with exact net and gross column conservation.
- Anatomical leaf-face illumination (`source_projection`, `beam_face_absorption`, `isotropic_face_absorption`, `regular_plate_passive`), sun/shade `PopulationQuadrature`, `LambertianFaces` / `MixedLambertianFaces` solid-angle quadrature, and `AngularEmissionBoundary` / `couple_emission`.
- Single- and dual-photosystem (`PSII + PSI`) conditional leaf spectral escape matrices (`LeafEmissionMatrix`, `DualLeafEmissionMatrix`, `FullEmissionSupport`, `build_leaf_emission_matrices`) with explicit out-of-window Stokes tail tracking.
- Spectral grid pullbacks and anchor bases (`IdentitySpectralMap`, `LinearSpectralMap`, `prepare_emission_basis`, `prepare_anchor_basis`).
- Hemispheric source projection (`LayerHemisphericSource`, `DualLayerHemisphericSource`, `evaluate_hemispheric_sif_bin`, `evaluate_hemispheric_sif_spectrum`), `KernelAbstractions` batch kernels, and `CanopyRadiationCUDAExt` (`adapt_radiation_storage`).
- Continuous-scalar `ForwardDiff.Dual` automatic differentiation support across conservative (`α → 0`), resonant (`k → m`), `LAI = 0`, and zenith/nadir (`|μ| = 1`) limits, plus `Documenter.jl` API reference and CI workflows.
