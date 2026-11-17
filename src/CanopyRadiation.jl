"""
    CanopyRadiation

Provide modular canopy-column radiative transfer, thermal and fluorescence
volume-emission cavity solvers, anatomical leaf-face illumination, conditional
single- and dual-photosystem leaf spectral escape matrices, spectral basis
pullbacks, and hemispheric source coupling on CPU and GPU.

`CanopyRadiation` builds directly on public `CanopyOptics` leaf and canopy
scattering types without depending on `vSmartMOM`, leaf physiology, or host
atmospheric state packages, preserving acyclic composition across `Float32`,
`Float64`, `OffsetArray`, `CuArray`, and `ForwardDiff` automatic
differentiation.

Source files, one per process or software boundary:

- `precision.jl`: scalar hooks and the precision policy; `records.jl`: equality,
  hashing, broadcasting and conversion shared by every record;
  `buffer_aliasing.jl`: output-buffer aliasing checks.
- `two_stream.jl`: two-stream layer rates and operators, passive column solvers;
  `emission_transport.jl`: volume emission, emitting layers and cavity solvers;
  `column_sweeps.jl`: the adding sweeps shared by the vector and batch solvers;
  `column_batch.jl`: `KernelAbstractions` batch column kernels.
- `leaf_face_absorption.jl`: beam and isotropic absorption by the leaf faces;
  `leaf_population.jl`: depth and leaf-orientation quadrature of leaf
  populations; `emission_angular_law.jl`: angular emission laws of the leaf faces;
  `population_emission.jl`: depth sources assembled from the face emission of a
  leaf population; `atmosphere_coupling.jl`: emission boundary coupled to a
  source-free atmosphere.
- `leaf_spectral_emission.jl`: leaf spectral escape matrices; `photon_units.jl`:
  photon bins, densities and energies per spectral row; `spectral_basis.jl`:
  spectral grid maps and excitation bases.
- `hemispheric_moments.jl`: hemisphere moments and their expansion into depth
  sources (scalar functions and batch kernels); `sif_assembly.jl`: layer
  hemispheric sources and canopy SIF evaluation; `device_storage.jl`: transfer of
  array-backed objects between host and device storage.
"""
module CanopyRadiation

using CanopyOptics:
    CanopyOptics,
    AbstractEmissionSupport,
    AbstractLeafBirthProfile,
    AbstractLeafDistribution,
    FourSAILGeometry,
    StrictStokes,
    TabulatedPhotonSpectrum,
    emission_probabilities,
    gauleg,
    leaf_emission_fates
using Distributions: pdf
using KernelAbstractions: KernelAbstractions, @index, @kernel
using LinearAlgebra: I, lu, mul!

import Base.Broadcast: broadcastable

# Scalar hooks for number types with derivative information (public, not exported)
@static if VERSION >= v"1.11"
    eval(
        Expr(
            :public,
            :unit_sin_from_cos,
            :primal_type,
            :primal,
            :all_finite,
            :exact_zero,
            :scalar_equal,
        ),
    )
end

include("precision.jl")
include("records.jl")
include("buffer_aliasing.jl")
include("two_stream.jl")
include("emission_transport.jl")
include("column_sweeps.jl")
include("column_batch.jl")
include("leaf_face_absorption.jl")
include("leaf_population.jl")
include("emission_angular_law.jl")
include("population_emission.jl")
include("atmosphere_coupling.jl")
include("leaf_spectral_emission.jl")
include("photon_units.jl")
include("spectral_basis.jl")
include("hemispheric_moments.jl")
include("sif_assembly.jl")
include("device_storage.jl")

export AbstractTwoStreamMethod,
    NativeTwoStream,
    AnalyticTwoStream,
    CanopyRates,
    foursail_canopy_rates,
    LayerOperator,
    diffuse_response,
    layer_operator,
    interior_flux,
    ColumnFluxWorkspace,
    solve_column!,
    solve_column
export AbstractVolumeEmission,
    NoEmission,
    UniformEmission,
    QuadratureEmission,
    EmittingLayer,
    point_emission_response,
    quadrature_emitting_layer,
    emitting_layer,
    EmittingColumnWorkspace,
    solve_emitting_column!,
    solve_emitting_column,
    solve_emission,
    point_emission_interior,
    emission_interior
export source_projection,
    beam_face_absorption,
    isotropic_face_absorption,
    regular_plate_passive,
    AbstractPopulationLayout,
    PairedIllumination,
    BlockedIllumination,
    PopulationQuadrature,
    population_coordinates,
    leaf_inclination_quadrature,
    orientation_hemisphere_samples,
    AbstractEmissionAngularLaw,
    LambertianFaces,
    MixedLambertianFaces,
    leaf_hemispheric_emission,
    leaf_angular_emission,
    depth_hemispheric_source,
    assemble_hemispheric_emission!,
    depth_angular_source,
    assemble_emission!,
    AngularEmissionBoundary,
    AbstractAtmosphericTransfer,
    VacuumAtmosphere,
    AngularAtmosphericTransfer,
    couple_emission,
    one_way_emission
export FullEmissionSupport,
    AbstractLeafEmission,
    NoLeafEmission,
    LeafEmissionMatrix,
    DualLeafEmissionMatrix,
    validate_leaf_emission,
    escape_leaf_emission!,
    escape_dual_leaf_emission!,
    build_leaf_emission_matrices,
    photon_bins!,
    photon_density!,
    photon_energy!
export AbstractSpectralMap,
    IdentitySpectralMap,
    LinearSpectralMap,
    source_grid,
    target_grid,
    map_spectrum!,
    source_basis_weights,
    prepare_emission_basis,
    validate_basis_amplitudes,
    prepare_anchor_basis,
    anchor_amplitudes
export hemispheric_moment,
    hemispheric_bin,
    hemispheric_source_moments!,
    expand_hemispheric_source!,
    LayerHemisphericSource,
    DualLayerHemisphericSource,
    evaluate_hemispheric_sif_bin,
    evaluate_hemispheric_sif_spectrum,
    batch_solve_columns!,
    batch_solve_emitting_columns!,
    batch_hemispheric_moments!,
    batch_expand_hemispheric_sources!,
    adapt_radiation_storage

end # module CanopyRadiation
