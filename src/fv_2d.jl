

function update_solution!(semi, dt)
	(; cache, boundary_conditions, grid) = semi
	(; u, du) = cache
	@. du = 0.0f0
	update_ghost_values!(cache, grid, boundary_conditions)
	compute_surface_fluxes!(semi)
	update_rhs!(semi)
	@. u -= dt * du
    apply_correction!(semi)

end

function update_ghost_values!(cache, grid::CartesianGrid2D, boundary_conditions::BoundaryConditions)

	apply_left_bc!(cache, boundary_conditions.left, grid.nx)
	apply_right_bc!(cache, boundary_conditions.right, grid.nx)
	apply_bottom_bc!(cache, boundary_conditions.bottom, grid.nz)
	apply_top_bc!(cache, boundary_conditions.top, grid.nz)

end

function apply_right_bc!(cache, right::PeriodicBC, nx)
	(; u) = cache
	u[:, nx+1, :] .= @views u[:, 2, :]
end

function apply_left_bc!(cache, left::PeriodicBC, nx)
	(; u) = cache
	u[:, 0, :] .= @views u[:, nx-1, :]
end

function apply_bottom_bc!(cache, bottom::PeriodicBC, nz)
	(; u) = cache
	u[:, :, 0] .= @views u[:, :, nz-1]
end

function apply_top_bc!(cache, top::PeriodicBC, nz)
	(; u) = cache
	u[:, :, nz+1] .= @views u[:, :, 2]
end

function compute_surface_fluxes!(semi)
	(; grid, equations, surface_flux, cache) = semi
	(; nx, nz) = grid
	(; u, fu) = cache

	nvar = Val(nvariables(equations))
	for i ∈ 1:nx+1
		for k ∈ 1:nz+1

			# x - direction
			fu[1, i, k, 1] = (u[1, i, k] + u[1, i-1, k])^2 * 0.25f0

			fu[2, i, k, 1] = (u[1, i, k] + u[1, i, k-1]) * (u[2, i, k] + u[2, i-1, k]) * 0.25f0

			# z - direction
			fu[1, i, k, 2] = (u[1, i, k] + u[1, i, k-1]) * (u[2, i, k] + u[2, i-1, k]) * 0.25f0

			fu[2, i, k, 2] = (u[2, i, k] + u[2, i, k-1])^2 * 0.25f0

		end
	end

end

function update_rhs!(semi)
	(; cache, grid, equations) = semi

	(; nx, nz, dx, dz) = grid
	(; fu, du) = cache
	nvar = Val(nvariables(equations))
	for i ∈ 1:nx
		for k ∈ 1:nz
			fn_rr = fu[:, i+1, k, 1]
			fn_ll = fu[:, i, k, 1]

			gn_rr = fu[:, i, k+1, 1]
			gn_ll = fu[:, i, k, 1]
			rhs = (fn_rr - fn_ll) / dx + (gn_rr - gn_ll) / dz
			du[:, i, k] .= rhs
		end
	end

end

function apply_correction!(semi)

    compute_div!(semi)
    compute_pressure!(semi)
    project_pressure!(semi)

end

function compute_div!(semi)

    (; cache, grid) = semi
    (; u, div) = cache
    (; nx,nz, dx, dz) = grid

    for i = 1:nx
        for k = 1:nz
            div[i,k] = (u[1,i+1,k] - u[1,i,k])/dx + (u[2,i,k+1] - u[2,i,k])/dz 
        end
    end

end

function compute_pressure!(semi)
    tol = 1e-8
    normres = 1
    om = 1.6
    (; cache, grid) = semi
    (; u, div) = cache
    (; dx, dz, nx, nz) = grid
    while normres > tol

        for i = 1:nx
            for k = 1:nz

            u[3,i,k] = u[3,i,k] + om*0.25*((u[3,i+1,k] -2*u[3,i,k]+ u[3,i-1,k])/dx^2 + (u[3,i,k-1] + u[3,i,k+1] - 2*u[3,i,k])/dz^2 -div[i,k]);
            normres = max(0.25*((u[3,i+1,k] -2*u[3,i,k]+ u[3,i-1,k])/dx^2 + (u[3,i,k-1] + u[3,i,k+1] - 2*u[3,i,k])/dz^2 -div[i,k]), normres)    
        end
        end
    end
end

# x - direction    
# ul = get_node_staggered(u, nvar, equations, i-1, j)
# ur = get_node_staggered(u, nvar, equations, i, j)

#flux = surface_flux(ul, ur, 1, equations)

# x - direction    
# ul = get_node_staggered(u, nvar, equations, i-1, j)
# ur = get_node_staggered(u, nvar, equations, i, j)

#flux = surface_flux(ul, ur, 1, equations)
#@. fu[1, i, k, 1] = flux[1]


# z - direction
#ul = get_node_vars(u, nvar, i-1)
#ur = get_node_vars(u, nvar, i)

#flux = surface_flux(ul, ur, 2, equations)    
#@. fu[:, i, k, 2] = flux

function get_node_staggered(u, nvar, equations::IncompressibleEuler2D, i, j)
	#incomplete
	return SVector(u[i, j], u[i+1, j-1])

end
