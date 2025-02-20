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

examples_dir() = joinpath(dirname(pathof(IncompressibleNavierStokes)), "..", "examples")

export mesh, SemiDiscretization, IncompressibleEuler2D

export flux_div

export SVector

export solve, ODE

export SORSolver, CGSolver, BiCGSTABSolver

export MyCPU

end
