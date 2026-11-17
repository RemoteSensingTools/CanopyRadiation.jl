# Passive & Emitting Column Solvers

Passive two-stream canopy layer operators, column workspaces, and unified volume-emission cavity solvers for longwave thermal radiation and solar-induced chlorophyll fluorescence.

## Passive Two-Stream Column Solvers

```@autodocs
Modules = [CanopyRadiation]
Pages = ["two_stream.jl"]
Order = [:type, :function]
```

## Emitting Cavity Solvers

```@autodocs
Modules = [CanopyRadiation]
Pages = ["emission_transport.jl"]
Order = [:type, :function]
```

## Batch Column Kernels

The batch kernels solve each column with the same adding sweep as
[`solve_column!`](@ref) and [`solve_emitting_column!`](@ref) (one sweep per column
kind, shared through column views), on the `KernelAbstractions` CPU backend or a GPU.

```@autodocs
Modules = [CanopyRadiation]
Pages = ["column_batch.jl"]
Order = [:type, :function]
```

## Precision Policy & Scalar Hooks

Every scalar that meets in a record or kernel of the column solvers must carry the
same floating-point type: `Float32` with `Float64`, integers, rationals,
irrationals and booleans throw an `ArgumentError`; a `Float64` with a
`Float64`-valued dual number promotes to the dual. Number types with derivative
information take part through five public (not exported) hooks, which a package
extension defines for its number type (`ForwardDiff.Dual`).

```@autodocs
Modules = [CanopyRadiation]
Pages = ["precision.jl"]
Order = [:function]
```
