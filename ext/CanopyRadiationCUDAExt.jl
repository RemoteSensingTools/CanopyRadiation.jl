module CanopyRadiationCUDAExt

using CUDA: CuArray
import CanopyRadiation: adapt_radiation_storage

adapt_radiation_storage(::Type{CuArray}, x::CuArray) = x
adapt_radiation_storage(::Type{CuArray}, x::AbstractArray) = CuArray(x)

end # module CanopyRadiationCUDAExt
