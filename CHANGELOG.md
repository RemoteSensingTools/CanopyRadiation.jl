# Changelog

All notable changes to `CanopyRadiation.jl` are documented in this file.

## [0.1.0] - 2026-11-12

### Added
- Passive two-stream canopy-column solvers (`NativeTwoStream`, `AnalyticTwoStream`), `CanopyRates`, `foursail_canopy_rates`, `LayerOperator`, and pre-allocated `ColumnFluxWorkspace` (`solve_column`, `solve_column!`).
- Unified longwave thermal and fluorescence cavity solvers (`NoEmission`, `UniformEmission`, `QuadratureEmission`, `EmittingLayer`, `EmittingColumnWorkspace`, `solve_emitting_column`, `solve_emitting_column!`, `solve_emission`) with exact net and gross column conservation.
- Anatomical leaf-face illumination (`source_projection`, `beam_face_absorption`, `isotropic_face_absorption`, `regular_plate_passive`), sun/shade `PopulationQuadrature`, `LambertianFaces` / `MixedLambertianFaces` solid-angle quadrature, and `AngularEmissionBoundary` / `couple_emission`.
- Single- and dual-photosystem (`PSII + PSI`) conditional leaf spectral escape matrices (`LeafEmissionMatrix`, `DualLeafEmissionMatrix`, `FullEmissionSupport`, `build_leaf_emission_matrices`) with explicit out-of-window Stokes tail tracking.
- Spectral grid pullbacks and anchor bases (`IdentitySpectralMap`, `LinearSpectralMap`, `prepare_emission_basis`, `prepare_anchor_basis`).
- Hemispheric source projection (`LayerHemisphericSource`, `DualLayerHemisphericSource`, `evaluate_hemispheric_sif_bin`, `evaluate_hemispheric_sif_spectrum`), `KernelAbstractions` batch kernels, and `CanopyRadiationCUDAExt` (`adapt_radiation_storage`).
- Continuous-scalar `ForwardDiff.Dual` automatic differentiation support across conservative (`α → 0`), resonant (`k → m`), `LAI = 0`, and zenith/nadir (`|μ| = 1`) limits, plus `Documenter.jl` API reference and CI workflows.
