"""
    SemiDiscretization

Struct containing everything about the spatial discretization, and the cache
used throughout the simulation.
"""
## Add Poisson Solver
struct SemiDiscretization{Grid, Equations <: AbstractEquations, SurfaceFlux, IC, BC,
                                    Solver, Cache}
    grid::Grid
    equations::Equations
    surface_flux::SurfaceFlux
    initial_condition::IC
    boundary_conditions::BC
    solver::Solver
    cache::Cache
end

"""
    SemiDiscretization(grid, equations, initial_condition, boundary_condition)

Constructor for the SemiDiscretizationHyperbolic struct to ensure periodic boundary conditions
are used by default.
"""
function SemiDiscretization(grid, equations, surface_flux, initial_condition;
    solver = FiniteVolumeSolver(),
    boundary_conditions = BoundaryConditions(PeriodicBC(), PeriodicBC(), PeriodicBC(), PeriodicBC()),
    cache = (;))

    cache = (;cache..., create_cache(equations, grid, initial_condition)...)

    SemiDiscretization(grid, equations, surface_flux, initial_condition, boundary_conditions, solver, cache)
end


"""
    create_cache(problem, grid)

Struct containing everything about the spatial discretization.
"""
function create_cache(equations, grid::CartesianGrid2D, initial_condition)

    (; xc, nx, nz, nbx, nbz) = grid
    RealT = eltype(xc)

    nvar = nvariables(equations)

    # Allocating variables
    # Conserved Variables
    u_ = zeros(RealT, nvar, nx + 2*nbx, nz + 2*nbz)
    u = OffsetArray(u_, OffsetArrays.Origin(1, 1-nbx, 1-nbz))

    # RHS Variables
    du_ = zeros(RealT, nvar, nx + 2*nbx, nz + 2*nbz)
    du = OffsetArray(du_, OffsetArrays.Origin(1, 1-nbx, 1-nbz))

    div_ = zeros(RealT, nx + 2*nbx, nz + 2*nbz)
    div = OffsetArray(div_, OffsetArrays.Origin(1-nbx, 1-nbz))

    orientations = 2
    fu_ = zeros(RealT, nvar, nx + 2*nbx, nz + 2*nbz, orientations)
    fu = OffsetArray(fu_, OffsetArrays.Origin(1, 1-nbx, 1-nbz, 1))
    initialize_variables!(u, grid, initial_condition, equations)

    cache = (; u, du, fu, div)

    return cache
end

function initialize_variables!(u, grid::CartesianGrid2D, initial_condition, equations)

    (; xc, zc, xf, zf, nx, nz) = grid

    for i = 1:nx
        for k = 1:nz
           u[:, i, k] = initial_condition((xc[i], zc[k]), (xf[i], zf[k]), 0.0, equations)
        end
    end

end