"""
    SemiDiscretization

Struct containing everything about the spatial discretization, and the cache
used throughout the simulation.
"""
## Add Poisson Solver
struct SemiDiscretization{Grid, Equations, SurfaceFlux, IC, BC,
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


struct var2D{ArrayType1}
    u::ArrayType1
    w::ArrayType1
    p::ArrayType1
end

"""
    create_cache(problem, grid)

Struct containing everything about the spatial discretization.
"""
function create_cache(equations, grid::CartesianGrid2D, initial_condition)

    (; xc, nx, nz, nbx, nbz) = grid
    RealT = eltype(xc)

    # Allocating variables
    u_ = zeros(RealT, nx + 2*nbx, nz + 2*nbz)
    u = OffsetArray(u_, OffsetArrays.Origin(1-nbx, 1-nbz))

    w_ = zeros(RealT, nx + 2*nbx, nz + 2*nbz)
    w = OffsetArray(w_, OffsetArrays.Origin(1-nbx, 1-nbz))
    
    p_ = zeros(RealT, nx + 2*nbx, nz + 2*nbz)
    p = OffsetArray(p_, OffsetArrays.Origin(1-nbx, 1-nbz))

    var = var2D(u, w, p)

    ## 2 because we are in 2D
    dMom_ = zeros(RealT, nx + 2*nbx, nz + 2*nbz, 2)
    dMom = OffsetArray(dMom_, OffsetArrays.Origin(1-nbx, 1-nbz, 0))

    dp_ = zeros(RealT, nx + 2*nbx, nz + 2*nbz)
    dp = OffsetArray(dp_, OffsetArrays.Origin(1-nbx, 1-nbz))

    ## Default 2D Taylor Green Vortex
    initialize_flow!(var, grid, initial_condition)

    cache = (; var, dMom, dp)

    return cache
end


function initialize_flow!(var, grid, initial_condition)

    (; xc, zc, xf, zf, nx, nz) = grid

    for i = 1:nx+1
        for k = 1:nz+1
            var.u[i,k] = -sinpi(xf[i]) * cospi(zc[k])
            var.w[i,k] =  cospi(xc[i]) * sinpi(zf[k])
            var.p[i,k] = 0.25f0*(cospi(2*xf[i])+sinpi(2*zf[k]))
        end
    end

end