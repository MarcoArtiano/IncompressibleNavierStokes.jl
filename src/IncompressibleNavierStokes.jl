module IncompressibleNavierStokes

using Trixi
using StaticArrays
using OffsetArrays

include("types.jl")
include("mesh.jl")
include("semi_discretization.jl")
include("IncompressibleEuler2D.jl")
include("time_integration.jl")
include("fv_2d.jl")
export mesh, SemiDiscretization, IncompressibleEuler2D

export flux_test

export SVector

end
