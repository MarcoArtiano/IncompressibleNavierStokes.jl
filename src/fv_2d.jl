import Trixi: get_node_vars

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
	u[:, nx+1, :] .= @views u[:, 1, :]
end

function apply_left_bc!(cache, left::PeriodicBC, nx)
	(; u) = cache
	u[:, 0, :] .= @views u[:, nx, :]
end

function apply_bottom_bc!(cache, bottom::PeriodicBC, nz)
	(; u) = cache
	u[:, :, 0] .= @views u[:, :, nz]
end

function apply_top_bc!(cache, top::PeriodicBC, nz)
	(; u) = cache
	u[:, :, nz+1] .= @views u[:, :, 1]
end

@inline function get_node_vars(u, ::Val{N}, indices...) where {N}
    SVector(ntuple(@inline(v->u[v, indices...]), N))
end


function compute_surface_fluxes!(semi)
	(; grid, equations, surface_flux, cache) = semi
	(; nx, nz) = grid
	(; u, fu) = cache

	nvar = Val(nvariables(equations))
	for i ∈ 1:nx+1
		for k ∈ 1:nz+1
			u_rr = get_node_vars(u, nvar, i, k)
			u_ll = get_node_vars(u, nvar, i-1, k)
			u_dd = get_node_vars(u, nvar, i, k-1)
			orientation = 1	
			flux = surface_flux(u_ll, u_rr, u_dd, orientation, equations)					
			fu[:, i, k, 1] .= flux

			orientation = 2			
			flux = surface_flux(u_dd, u_rr, u_ll, orientation, equations)
			fu[:, i, k, 2] .= flux

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
			orientation = 1
			fn_rr = get_node_vars(fu, nvar, i+1, k, orientation) 
			fn_ll = get_node_vars(fu, nvar, i, k, orientation) 

			orientation = 2
			gn_rr = get_node_vars(fu, nvar, i, k+1, orientation) 
			gn_ll = get_node_vars(fu, nvar, i, k, orientation) 
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
    tol = 1e-13
    normres = 1
    om = 1.6
    (; cache, grid, boundary_conditions) = semi
    (; u, div, normatrix) = cache
    (; dx, dz, nx, nz) = grid
    #add max iter
    while normres > tol
        # That doesn't not support non-uniform mesh.
        for i = 1:nx
            for k = 1:nz
            u[3,i,k] = u[3,i,k] + om*0.25*(u[3,i+1,k] -2*u[3,i,k]+ u[3,i-1,k] + u[3,i,k-1] + u[3,i,k+1] - 2*u[3,i,k] -dx*dz*div[i,k])
            end
        end
        # This is not efficient
	    update_ghost_values!(cache, grid, boundary_conditions)

        # This is not efficient
        for i = 1:nx
            for k = 1:nz
        normatrix[i,k] = abs((u[3,i+1,k] -2*u[3,i,k]+ u[3,i-1,k])/dx^2 + (u[3,i,k-1] + u[3,i,k+1] - 2*u[3,i,k])/dz^2 -div[i,k])    
            end
        end
        normres = maximum(normatrix)
        @show normres
    end
end

function project_pressure!(semi)

    (; cache, grid) = semi
    (; u, div, normatrix) = cache
    (; dx, dz, nx, nz) = grid

    for i = 1:nx
        for k = 1:nz
            u[1,i,k] = u[1,i,k] -(u[3,i,k] - u[3,i-1,k])/dx
            u[2,i,k] = u[2,i,k] -(u[3,i,k] - u[3,i,k-1])/dz
        end
    end

end

