module IncompressibleNavierStokes

using StaticArrays
using OffsetArrays

include("types.jl")
include("mesh.jl")
include("semi_discretization.jl")

export mesh, SemiDiscretization

end
