using IncompressibleNavierStokes
## IncompressibleEuler2D(gamma, rho)
equations = IncompressibleEuler2D(1.4, 1.0)

function initial_condition_tgv(x, xf, t, equations::IncompressibleEuler2D)
    # xf are the nodes values
    # x are the center values
    u = -sinpi(xf[1]) * cospi(x[2])
    w =  cospi(x[1]) * sinpi(xf[2])
    p = 0.25f0*(cospi(2*xf[1])+sinpi(2*xf[2]))

    return SVector(u, w, p)
end

nx = 29
nz = 29
domain = (0.0, 2.0, 0.0, 2.0)
grid = IncompressibleNavierStokes.mesh(domain, nx, nz)
surface_flux = flux_div
semi = SemiDiscretization(grid, equations, surface_flux, initial_condition_tgv;
                          matrix_solver = SORSolver(maxiter = 1000, tol = 1e-12, om = 1.6))

dt =  6e-4

tspan = (0.0, 12.0)

ode = ODE(semi, tspan)

sol = solve(ode, dt);
