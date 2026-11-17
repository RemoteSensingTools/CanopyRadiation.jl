# Hemispheric SIF, Spectral Bases, & Kernels

Conservative spectral grid pullbacks, anchor bases, hemispheric source projection, multi-bin canopy SIF transport, `KernelAbstractions` batch kernels, and CUDA storage adaptation.

## Spectral Grid Pullbacks & Anchor Bases

```@autodocs
Modules = [CanopyRadiation]
Pages = ["spectral_basis.jl"]
Order = [:type, :function]
```

## Hemispheric Moments & Source Expansion (Scalar and Batch Kernels)

```@autodocs
Modules = [CanopyRadiation]
Pages = ["hemispheric_moments.jl"]
Order = [:type, :function]
```

## Layer Hemispheric Sources & SIF Assembly

```@autodocs
Modules = [CanopyRadiation]
Pages = ["sif_assembly.jl"]
Order = [:type, :function]
```

## Device Storage Adaptation

```@autodocs
Modules = [CanopyRadiation]
Pages = ["device_storage.jl"]
Order = [:type, :function]
Private = false
```
