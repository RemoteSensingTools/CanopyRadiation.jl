# Leaf-Face Illumination & Spectral Escape

Anatomical adaxial/abaxial leaf-face absorption, sun/shade leaf inclination quadrature, directional emission boundary coupling, and single- or dual-photosystem (`PSII + PSI`) conditional spectral escape matrices.

## Leaf-Face Absorption

```@autodocs
Modules = [CanopyRadiation]
Pages = ["leaf_face_absorption.jl"]
Order = [:type, :function]
Private = false
```

## Leaf Populations & Orientation Quadrature

```@autodocs
Modules = [CanopyRadiation]
Pages = ["leaf_population.jl"]
Order = [:type, :function]
Private = false
```

## Leaf Emission Angular Laws & Depth Sources

```@autodocs
Modules = [CanopyRadiation]
Pages = ["emission_angular_law.jl"]
Order = [:type, :function]
Private = false
```

## Canopy-Atmosphere Emission Coupling

```@autodocs
Modules = [CanopyRadiation]
Pages = ["atmosphere_coupling.jl"]
Order = [:type, :function]
Private = false
```

```@docs
CanopyRadiation.unit_sin_from_cos
```

## Conditional Leaf Spectral Escape Matrices

```@autodocs
Modules = [CanopyRadiation]
Pages = ["leaf_spectral_emission.jl"]
Order = [:type, :function]
```
