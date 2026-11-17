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
include("scalars.jl")
include("two_stream.jl")
include("emission_transport.jl")
include("column_sweeps.jl")
include("column_batch.jl")
include("face_and_geometry.jl")
include("leaf_spectral_emission.jl")
include("spectral_basis.jl")
include("hemispheric_sources.jl")

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
