using IncompressibleNavierStokes
using CairoMakie
## IncompressibleEuler2D(gamma, rho)
equations = IncompressibleEuler2D(1.4, 1.0)

function initial_conditoin_tgv(x, xf, t, equations::IncompressibleEuler2D)

    u = -sinpi(xf[1]) * cospi(x[2])
    w =  cospi(x[1]) * sinpi(xf[2])
    p = 0.25f0*(cospi(2*x[1])+sinpi(2*xf[2]))

    return SVector(u, w, p)
end

nx = 64
nz = 64
domain = (0.0, 2.0, 0.0, 2.0)
grid = IncompressibleNavierStokes.mesh(domain, nx, nz)
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

    fig = Figure()
    ax = Axis(fig[1,1], title="divergence")
    
    # Plottiamo il contourf (filled contour)
    hm = contourf!(ax,grid.xf[1:grid.nx], grid.zc[1:grid.nz], semi.cache.div[1:grid.nx,1:grid.nz])
    
    # Aggiungiamo la colorbar
    Colorbar(fig[1,2], hm)
    display(fig)
    throw(error)
    
    IncompressibleNavierStokes.compute_pressure!(semi)
    #contourf(grid.xf[1:grid.nx], grid.zc[1:grid.nz], sol[1,1:grid.nx,1:grid.nz])
