using IncompressibleNavierStokes
using Metal
using KernelAbstractions

RealT = Float64

gamma, rho = map(RealT, (1.4, 1.0))

## IncompressibleEuler2D(gamma, rho)
equations = IncompressibleEuler2D(gamma, rho)

function initial_condition_tgv(x, xf, t, equations::IncompressibleEuler2D)
    # xf are the nodes values
    # x are the center values
    u = -sinpi(xf[1]) * cospi(x[2])
    w =  cospi(x[1]) * sinpi(xf[2])
    p = 0.25f0*(cospi(2.0f0*xf[1])+sinpi(2.0f0*xf[2]))

    return SVector(u, w, p)
end

nx = 29
nz = 29
domain = map(RealT, (0.0, 2.0, 0.0, 2.0))
grid = IncompressibleNavierStokes.mesh(domain, nx, nz, backend = CPU());
surface_flux = flux_div
semi = SemiDiscretization(grid, equations, surface_flux, initial_condition_tgv;
                        #   matrix_solver = BiCGSTABSolver(maxiter = 1000, tol = map(RealT, 1e-6)),
                          matrix_solver = CGSolver(maxiter = 1000, tol = map(RealT, 0.5e-5)),
                          backend = CPU()
                          )

dt =  6e-4

tspan = (0.0, 11.0)

ode = ODE(semi, tspan)

sol = solve(ode, dt);
