# Leaf-Face Illumination & Spectral Escape

Anatomical adaxial/abaxial leaf-face absorption, sun/shade leaf inclination quadrature, directional emission boundary coupling, and single- or dual-photosystem (`PSII + PSI`) conditional spectral escape matrices.

## Leaf-Face Illumination, Population Geometry, & Angular Escape

```@autodocs
Modules = [CanopyRadiation]
Pages = ["face_and_geometry.jl"]
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
