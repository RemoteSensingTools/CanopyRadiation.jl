# CanopyRadiation.jl

Modular canopy-column radiative transfer, thermal and fluorescence cavity transport, anatomical leaf-face illumination, conditional single- and dual-photosystem (`PSII + PSI`) leaf spectral escape, spectral basis pullbacks, and hemispheric source coupling in Julia.

## Architecture and Dependency Boundary

`CanopyRadiation.jl` sits directly above `CanopyOptics.jl` (`v0.2.3`) with zero dependency on `vSmartMOM`, leaf physiology (`LeafPhotosynthesis.jl`), canopy scalar exchange (`CanopyExchange.jl`), or atmospheric host state packages (`Thermodynamics.jl`, `ClimaCore.jl`):

- **Passive Two-Stream Column Solvers:** [`NativeTwoStream`](@ref) (thin-slab Taylor direct source + adding-doubling) and [`AnalyticTwoStream`](@ref) (closed-form `r_∞`-free Green's-function operator), positional `@.`-broadcastable [`foursail_canopy_rates`](@ref), and zero-scratch [`solve_column`](@ref) / [`solve_column!`](@ref) ([`ColumnFluxWorkspace`](@ref)).
- **Unified Thermal & Fluorescence Cavity Solvers:** [`NoEmission`](@ref), [`UniformEmission`](@ref), [`QuadratureEmission`](@ref), [`emitting_layer`](@ref), 7-field [`solve_emitting_column`](@ref) / [`solve_emitting_column!`](@ref) ([`EmittingColumnWorkspace`](@ref)) satisfying net and gross column conservation, and zero-incident [`solve_emission`](@ref).
- **Leaf-Face Illumination, Population Geometry, & Angular Escape:** [`source_projection`](@ref), [`beam_face_absorption`](@ref), [`isotropic_face_absorption`](@ref), [`regular_plate_passive`](@ref), [`PopulationQuadrature`](@ref), [`leaf_inclination_quadrature`](@ref), [`orientation_hemisphere_samples`](@ref), [`LambertianFaces`](@ref), [`MixedLambertianFaces`](@ref), and [`AngularEmissionBoundary`](@ref) / [`couple_emission`](@ref).
- **Single- & Dual-Photosystem Conditional Spectral Escape:** [`LeafEmissionMatrix`](@ref) (with explicit `out_of_window` tail ledger), [`DualLeafEmissionMatrix`](@ref), [`FullEmissionSupport`](@ref), full-spectrum `window_indices` [`build_leaf_emission_matrices`](@ref) (avoiding false Stokes renormalization), [`prepare_emission_basis`](@ref), and [`prepare_anchor_basis`](@ref).
- **Hemispheric Source Projection & GPU Batch Kernels:** [`LayerHemisphericSource`](@ref) (unit-interval `fractional_depths`), [`DualLayerHemisphericSource`](@ref), [`evaluate_hemispheric_sif_bin`](@ref), [`evaluate_hemispheric_sif_spectrum`](@ref), `KernelAbstractions` batch kernels ([`batch_solve_columns!`](@ref), [`batch_solve_emitting_columns!`](@ref), [`batch_hemispheric_moments!`](@ref), [`batch_expand_hemispheric_sources!`](@ref)), and `CanopyRadiationCUDAExt` ([`adapt_radiation_storage`](@ref)).

## Quickstart

```jldoctest
julia> using CanopyRadiation

julia> rates = CanopyRates(0.10, 0.10, 0.05, 0.05, 0.50);

julia> op = layer_operator(AnalyticTwoStream(), rates, 1.5);

julia> col = solve_column([op, op], 0.8, 0.2, 0.15);

julia> isapprox(0.8 + 0.2, col.up[1] + sum(col.absorption) + col.ground; atol = 1e-14)
true
```

## API Organization

- [Overview & Index](api/index.md): Module docstring and full alphabetical index.
- [Passive & Emitting Column Solvers](api/column_and_cavity.md): Two-stream operators, adding-doubling column solvers, and emitting cavity solvers.
- [Leaf-Face Illumination & Spectral Escape](api/faces_and_emission.md): Anatomical leaf-face irradiance, sun/shade population geometry, angular escape, and conditional `PSII + PSI` emission matrices.
- [Hemispheric SIF, Spectral Bases, & Kernels](api/hemispheric_and_kernels.md): Spectral grid maps, hemispheric SIF transport, `KernelAbstractions` batch kernels, and CUDA adaptation.
