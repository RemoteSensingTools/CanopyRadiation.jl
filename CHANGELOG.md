# Changelog

All notable changes to `CanopyRadiation.jl` are documented in this file.

## Unreleased

Internal refactor of the whole package. No exported name was removed or renamed, no
exported argument order or return field changed, and every output of the frozen
v0.1.0 numerical records (`test/frozen_record`: core, remaining public API,
CanopyOptics-dependent tables) is bit-identical on the tuple, vector and batch
paths. Because of the precision policy and the scalar hooks below the release is
**breaking** (not a patch release). The version in `Project.toml` is not bumped.

### Added
- Package extension `CanopyRadiationForwardDiffExt` (weak dependency ForwardDiff
  0.10 or 1): defines the scalar hooks for `ForwardDiff.Dual`, so AD behaves
  identically on both ForwardDiff majors.

### Breaking
- Number types other than plain floating-point numbers take part only through five
  public, not exported, scalar hooks: `CanopyRadiation.primal_type`, `primal`,
  `all_finite`, `exact_zero` and `scalar_equal` (defined for `ForwardDiff.Dual` by
  the extension). v0.1.0 accepted any `Real` struct whose first field was a float;
  such a type is now rejected until it defines the hooks. The reflection helpers of
  the deleted `scalars.jl` (`primal_float_type`, `primal_scalar`,
  `is_continuous_scalar`, `compatible_scalar`, `promote_compatible_scalar`,
  `require_continuous_scalar`, `finite_scalar`, `exact_zero_scalar`) are gone.
- One precision policy (`src/precision.jl`). Where a function takes its scalars,
  records or arrays as independent arguments, mixed `Float32`/`Float64` and
  integer, rational, irrational or boolean values (or element types), and abstract
  element types such as `AbstractFloat[...]`, throw an `ArgumentError` with a
  static message (v0.1.0: `MethodError`). This holds for `CanopyRates`,
  `LayerOperator`, `NativeTwoStream`, `UniformEmission`, `EmittingLayer`,
  `foursail_canopy_rates`, `diffuse_response`, `layer_operator`, `interior_flux`,
  `point_emission_response`, `point_emission_interior`, `emitting_layer`,
  `quadrature_emitting_layer`, `emission_interior`, `solve_column`,
  `solve_column!`, `solve_emitting_column`, `solve_emitting_column!`,
  `solve_emission`, `unit_sin_from_cos`, `source_projection`,
  `beam_face_absorption`, `isotropic_face_absorption`, `regular_plate_passive`,
  `MixedLambertianFaces`, `leaf_hemispheric_emission`, `leaf_angular_emission`,
  `depth_hemispheric_source`, `depth_angular_source`, `leaf_inclination_quadrature`,
  `orientation_hemisphere_samples`, `couple_emission`, `one_way_emission`,
  `hemispheric_moment`, `hemispheric_bin`, `evaluate_hemispheric_sif_bin`,
  `evaluate_hemispheric_sif_spectrum`, `convert` between records, the
  `ColumnFluxWorkspace(FT, n)` and `EmittingColumnWorkspace(FT, n)` constructors,
  and the inputs of `hemispheric_source_moments!`, `expand_hemispheric_source!`,
  `batch_hemispheric_moments!`, `batch_expand_hemispheric_sources!`,
  `assemble_hemispheric_emission!` and `assemble_emission!` (including an output
  element type that differs from the promoted input type). A `Float64` mixed with a
  `Float64`-valued dual number still promotes to the dual.
- Where one type parameter `{FT}` is shared by several arguments of the signature,
  mixed precision is still a `MethodError` (no matching method), as in v0.1.0, and
  only a non-float or abstract `FT` gives the `ArgumentError`:
  `batch_solve_columns!`, `batch_solve_emitting_columns!`, `photon_bins!`,
  `photon_density!`, `photon_energy!` (including its coefficient),
  `escape_leaf_emission!` (model versus buffers), the output pairs of the
  hemisphere drivers and of `assemble_hemispheric_emission!`, the constructors of
  `QuadratureEmission`, `LayerHemisphericSource`, `PopulationQuadrature`,
  `LeafEmissionMatrix`, `AngularEmissionBoundary`, `AngularAtmosphericTransfer`,
  `ColumnFluxWorkspace` and `EmittingColumnWorkspace` from arrays, and the
  `AbstractFloat`-only spectral functions (`IdentitySpectralMap`,
  `LinearSpectralMap`, `prepare_emission_basis`, `validate_basis_amplitudes`,
  `prepare_anchor_basis`, `anchor_amplitudes`, `build_leaf_emission_matrices`).
- `escape_leaf_emission!` and `escape_dual_leaf_emission!` with `NoLeafEmission`
  reject integer and rational output buffers (v0.1.0 filled them with zeros).
- A `DualLeafEmissionMatrix` must hold a `LeafEmissionMatrix` or `NoLeafEmission`
  in each photosystem; a nested `DualLeafEmissionMatrix` throws an `ArgumentError`
  (v0.1.0 silently treated it as no emission).
- `solve_column!` and `solve_emitting_column!` throw a `DimensionMismatch` when a
  workspace vector was resized after construction (v0.1.0 wrote out of bounds).

### Changed
- Every record (two-stream methods, rates and operators, volume emissions, emitting
  layers, population quadrature and layouts, emission angular laws, emission
  boundary and atmospheres, leaf emission models, spectral maps, layer hemispheric
  sources) shares one definition of `==`, `isequal`, `hash` and broadcasting
  (`src/records.jl`), including records that hold host or device arrays: `-0.0 ==
  0.0`, `NaN != NaN`, `isequal` and `hash` consistent, array axes part of the value,
  and derivative components compared on both ForwardDiff majors (v0.1.0: `==`
  followed the ForwardDiff version; this also applies to the LAI check of
  `DualLayerHemisphericSource`). Host arrays are compared in place without
  allocating; device arrays through one host copy. Hash values differ from v0.1.0.
  The exported abstract types are subtypes of the internal record supertype.
  `FullEmissionSupport` (a CanopyOptics subtype) keeps Base equality.
- Records broadcast as scalars. For records that hold arrays
  (`QuadratureEmission`, `PopulationQuadrature`, the leaf emission models, the
  emission boundary and atmosphere, the spectral maps, the layer hemispheric
  sources) this works on the CPU only: they are not isbits, so they cannot enter a
  broadcast over GPU arrays.
- Dispatch replaces type switches: method parameters in `layer_operator` and
  `interior_flux`; photosystem pairs of `DualLeafEmissionMatrix` (validation,
  escape, bin checks, bin axis, out-of-window ledger, layer/model compatibility);
  the forms of `out_of_window_fraction` and the `StrictStokes` tail check; the CPU
  fast paths of `photon_bins!`, `photon_density!`, `photon_energy!`,
  `hemispheric_source_moments!` and `expand_hemispheric_source!` are methods on the
  backend type (the three photon conversions share one kernel).
- One implementation per algorithm: the passive and the emitting column each have
  one adding sweep (`src/column_sweeps.jl`) shared by `solve_column!` /
  `solve_emitting_column!` and the batch kernels (`src/column_batch.jl`); the
  hemisphere moment and bin reductions are each written once and shared by the
  scalar functions, the driver kernels and the batch kernels; the tuple recursions
  stay.
- `adapt_radiation_storage` has one field-mapping method for every array-holding
  object instead of eleven (same objects, same results, `QuadratureEmission` and
  `LayerHemisphericSource` still rebuilt without value checks).
- Source files by process: `face_and_geometry.jl` became `leaf_face_absorption.jl`,
  `leaf_population.jl`, `emission_angular_law.jl`, `population_emission.jl` and
  `atmosphere_coupling.jl`; `hemispheric_sources.jl` became
  `hemispheric_moments.jl`, `sif_assembly.jl` and `device_storage.jl`; the photon
  unit conversions moved to `photon_units.jl`; `buffer_aliasing.jl` holds the output
  aliasing checks; `scalars.jl` is deleted (including the unused `neumaier_add`).

### Known, unchanged
Kept on purpose; each changes behaviour and needs a decision:
- `batch_hemispheric_moments!` and `batch_expand_hemispheric_sources!` validate less
  than their scalar counterparts (no cosine range, finiteness or sign checks).
- Validation tolerances differ: `validate_leaf_emission(::DualLeafEmissionMatrix)`
  defaults to `atol = 1e-6`, the single-matrix method to `sqrt(eps)`. In `Float32`
  `1e-6` (about 8 eps) is 345 times tighter than the single-matrix default and
  rejects exactly normalised matrices increasingly above about 400 spectral bins
  (1 % at 400, 7 % at 601, 56 % at 1,201); in `Float64` it is 67 times looser.
- With dual numbers `diffuse_response` and the native thin slab use the small-q
  series where floats use the closed form. The value of a dual result differs from
  the plain-float result by at most about 4.8 eps for `diffuse_response`; for
  `NativeTwoStream` the difference is amplified by the doublings, about `2^d`:
  24 eps at `d = 4`, 450 at `d = 8`, `1e4` at `d = 16`.
- `NativeTwoStream` accuracy degrades with the number of doublings `d`, which is set
  by `max(a + b, k)`, so a low sun degrades the diffuse operator as well: for
  conservative scattering `|1 - R - T|` is about `1.2 × 2^d eps`, in `Float64` at
  most about `1.5e-11` (`d = 16`); in `Float32` `2e-5` at `d = 7` (LAI 1, solar
  zenith 85°) and `1.3e-3` at `d = 13` (LAI 1, solar zenith 89.9°), with negative
  layer absorption of the same size. Use `AnalyticTwoStream` for `Float32` at low
  sun. `layer_operator(NativeTwoStream(...), rates, Inf)` does not terminate.
- The passive tuple and vector paths agree bit for bit for one or more layers; for
  a column of zero layers the tuple method computes `soil * (direct + diffuse)` and
  the vector and batch paths `soil * diffuse + soil * direct`, which differ by up to
  2 ulp (about 31 % of random inputs).
- Zero-LAI layers pass fluxes through exactly, but their absorption inside a column
  is a flux difference and can be ±3.5 eps of the incident flux.
- Physically invalid inputs are accepted silently, as before: negative or NaN LAI
  and rates, soil reflectance outside [0, 1], negative `UniformEmission`, |cos| > 1
  in `leaf_hemispheric_emission` and `PopulationQuadrature`, zero spectral bin
  width.
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
