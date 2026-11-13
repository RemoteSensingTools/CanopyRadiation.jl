# CanopyRadiation.jl

Modular canopy-column radiative transfer, thermal and fluorescence cavity transport, anatomical leaf-face illumination, conditional single- and dual-photosystem (`PSII + PSI`) leaf spectral escape, spectral basis pullbacks, and hemispheric source coupling in Julia.

## Architecture and Dependency Boundary

`CanopyRadiation.jl` sits directly above `CanopyOptics.jl` (`v0.2.3`) with zero dependency on `vSmartMOM`, leaf physiology (`LeafPhotosynthesis.jl`), canopy scalar exchange (`CanopyExchange.jl`), or atmospheric host state packages (`Thermodynamics.jl`, `ClimaCore.jl`):

- **Passive Two-Stream Column Solvers:** `NativeTwoStream` (thin-slab Taylor direct source + adding-doubling) and `AnalyticTwoStream` (closed-form `r_∞`-free Green's-function operator), positional `@.`-broadcastable `foursail_canopy_rates`, and zero-scratch `solve_column` / `solve_column!` (`ColumnFluxWorkspace`).
- **Unified Thermal & Fluorescence Cavity Solvers:** `NoEmission`, `UniformEmission`, `QuadratureEmission`, `emitting_layer`, 7-field `solve_emitting_column` / `solve_emitting_column!` (`EmittingColumnWorkspace`) satisfying net and gross column conservation, and zero-incident `solve_emission`.
- **Leaf-Face Illumination, Population Geometry, & Angular Escape:** `source_projection` (with zenith/nadir tangent-safe `unit_sin_from_cos`), `beam_face_absorption`, `isotropic_face_absorption`, `regular_plate_passive`, `PopulationQuadrature`, `leaf_inclination_quadrature`, `orientation_hemisphere_samples`, `LambertianFaces`, `MixedLambertianFaces`, and `AngularEmissionBoundary` / `couple_emission`.
- **Single- & Dual-Photosystem Conditional Spectral Escape:** `LeafEmissionMatrix` (with explicit `out_of_window` tail ledger), `DualLeafEmissionMatrix`, `FullEmissionSupport`, full-spectrum `window_indices` `build_leaf_emission_matrices` (avoiding false Stokes renormalization), `prepare_emission_basis`, and `prepare_anchor_basis`.
- **Hemispheric Source Projection & GPU Batch Kernels:** `LayerHemisphericSource` (unit-interval `fractional_depths`), `DualLayerHemisphericSource`, `evaluate_hemispheric_sif_bin`, `evaluate_hemispheric_sif_spectrum`, `KernelAbstractions` batch kernels (`batch_solve_columns!`, `batch_solve_emitting_columns!`, `batch_hemispheric_moments!`, `batch_expand_hemispheric_sources!`), and `CanopyRadiationCUDAExt` (`adapt_radiation_storage`).

## Testing

```bash
julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()'
```

Optional NVIDIA GPU verification (`CUDA.allowscalar(false)`):

```bash
julia --startup-file=no -e '
using Pkg; Pkg.activate(temp=true)
Pkg.develop(path="."); Pkg.add(["CUDA", "Aqua", "Test"])
include("test/run_cuda.jl")
'
```
