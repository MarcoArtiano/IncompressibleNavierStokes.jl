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
    matrix_solver::AbstractMatrixSolver
    cache::Cache
end

"""
    SemiDiscretization(grid, equations, initial_condition, boundary_condition)

Constructor for the SemiDiscretizationHyperbolic struct to ensure periodic boundary conditions
are used by default.
"""
function SemiDiscretization(grid, equations, surface_flux, initial_condition;
    solver = FiniteVolumeSolver(),
    boundary_conditions = BoundaryConditions(PeriodicBC(), PeriodicBC(), PeriodicBC(),
                                             PeriodicBC()),
    matrix_solver = SORSolver(), backend = MyCPU(), cache = (;))

    cache = (;cache..., create_cache(equations, grid, initial_condition, backend)...)

    SemiDiscretization(grid, equations, surface_flux, initial_condition, boundary_conditions,
                       solver, matrix_solver, cache)
end

"""
    create_cache(problem, grid)

Struct containing everything about the spatial discretization.
"""
function create_cache(equations, grid::CartesianGrid2D, initial_condition, backend)

    (; xc, nx, nz, nbx, nbz) = grid
    RealT = eltype(xc)

    nvar = nvariables(equations)

    # Allocating variables
    # Conserved variables
    u_ = KernelAbstractions.zeros(backend, RealT, nvar, nx + 2*nbx, nz + 2*nbz)
    u = OffsetArray(u_, OffsetArrays.Origin(1, 1-nbx, 1-nbz))

    # RHS variables
    du_ = KernelAbstractions.zeros(backend, RealT, nvar, nx + 2*nbx, nz + 2*nbz)
    du = OffsetArray(du_, OffsetArrays.Origin(1, 1-nbx, 1-nbz))

    div_ = KernelAbstractions.zeros(backend, RealT, nx + 2*nbx, nz + 2*nbz)
    div = OffsetArray(div_, OffsetArrays.Origin(1-nbx, 1-nbz))
    # TODO: use Trixi function for the dimension
    dimensions = 2
    fu_ = KernelAbstractions.zeros(backend, RealT, nvar, nx + 2*nbx, nz + 2*nbz, dimensions)
    fu = OffsetArray(fu_, OffsetArrays.Origin(1, 1-nbx, 1-nbz, 1))

    initialize_variables!(u, grid, initial_condition, equations, backend)

    normatrix = zeros(RealT, nx, nz)

    cache = (; u, du, fu, div, normatrix)

    return cache
end

function initialize_variables!(u, grid::CartesianGrid2D, initial_condition, equations,
                               backend::MyCPU
                               )

    (; xc, zc, xf, zf, nx, nz) = grid

    for i = 1:nx
        for k = 1:nz
           u[:, i, k] = initial_condition((xc[i], zc[k]), (xf[i], zf[k]), 0.0, equations)
        end
    end
end

function initialize_variables!(u, grid::CartesianGrid2D, initial_condition, equations,
                               backend::Union{GPU, CPU})

    (; xc, zc, xf, zf, nx, nz) = grid

    nvar = Val(nvariables(equations))

    initialize_variables_kernel!(backend)(u, xc, zc, xf, zf, nvar, equations, initial_condition, zero(eltype(u)),
                                          ndrange = (nx, nz))

    # for i = 1:nx
    #     for k = 1:nz
    #        u[:, i, k] = initial_condition((xc[i], zc[k]), (xf[i], zf[k]), 0.0, equations)
    #     end
    # end
end

@kernel function initialize_variables_kernel!(u, xc, zc, xf, zf, ::Val{N}, equations, initial_condition, t) where {N}
    i, k = @index(Global, NTuple)
    ic = initial_condition((xc[i], zc[k]), (xf[i], zf[k]), t, equations)
    for n=1:N
        @inbounds u[n, i, k] = ic[n]
    end
end

