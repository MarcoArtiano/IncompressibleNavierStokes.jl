using IncompressibleNavierStokes
## IncompressibleEuler2D(gamma, rho)
equations = IncompressibleEuler2D(1.4, 1.0)

function initial_conditoin_tgv(x, xf, t, equations::IncompressibleEuler2D)

    u = -sinpi(xf[1]) * cospi(x[2])
    w =  cospi(x[1]) * sinpi(xf[2])
    p = 0.25f0*(cospi(2*x[1])+sinpi(2*xf[2]))

    return SVector(u, w, p)
end

nx = 10
nz = 10
domain = (0.0, 1.0, 0.0, 2.0)
grid = mesh(domain, nx, nz)
surface_flux = flux_test
semi = SemiDiscretization(grid, equations, flux_test, initial_conditoin_tgv)
dt = 0.001
(; cache, boundary_conditions, grid) = semi
	(; u, du) = cache
	@. du = 0.0f0
	IncompressibleNavierStokes.update_ghost_values!(cache, grid, boundary_conditions)
	IncompressibleNavierStokes.compute_surface_fluxes!(semi)
	IncompressibleNavierStokes.update_rhs!(semi)
	@. u -= dt * du

    IncompressibleNavierStokes.compute_div!(semi)
    IncompressibleNavierStokes.compute_pressure!(semi)
    #contourf(grid.xf[1:grid.nx], grid.zc[1:grid.nz], sol[1,1:grid.nx,1:grid.nz])
