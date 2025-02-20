import Trixi: get_node_vars
using ConjugateGradientsGPU

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

# TODO: to add a struct for Poisson solver: div flux, p flux, poisson parameters

function apply_correction!(semi)

    compute_div!(semi)
    compute_pressure!(semi, semi.matrix_solver)
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

function compute_pressure!(semi, matrix_solver::SORSolver)
    # TODO: Move into a struct:
	@unpack tol, maxiter, om = matrix_solver
    normres = 1
    (; cache, grid, boundary_conditions) = semi
    (; u, div, normatrix) = cache
    (; dx, dz, nx, nz) = grid
    #add max iter
	it = 0
    while normres > tol
		if it > max_iter
			println("Max iter reached")
			break
		end
		it += 1
        # That doesn't support non-uniform mesh.
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
    end
end

function laplace_2d!(x, u, nx, nz, dx, dz)
	x_mat = reshape(x, nx, nz)
	u_mat = reshape(u, nx, nz)

	for i = 1:nx
		im1 = (i == 1)  ? nx : i - 1  # Wrap around for periodic BC
		ip1 = (i == nx) ? 1  : i + 1  # Wrap around for periodic BC

		for k = 1:nz
			km1 = (k == 1)  ? nz : k - 1  # Wrap around for periodic BC
			kp1 = (k == nz) ? 1  : k + 1  # Wrap around for periodic BC

			x_mat[i, k]  = (u_mat[ip1, k] - 2.0f0 * u_mat[i, k] + u_mat[im1, k]) / dx^2  # x direction
			x_mat[i, k] += (u_mat[i, km1] - 2.0f0 * u_mat[i, k] + u_mat[i, kp1]) / dz^2  # z direction
		end
	end

	x .*= -1.0
end

function compute_pressure!(semi, matrix_solver::CGSolver)
	(; cache, grid, boundary_conditions) = semi
	(; div) = cache
	(;dx, dz, nx, nz) = grid
	update_ghost_values!(cache, grid, boundary_conditions)

	@unpack tol, maxiter = matrix_solver
	u_new, exit_code, num_iters = cg(
		(x,u) -> laplace_2d!(x, u, nx, nz, dx, dz), vec(div[1:nx, 1:nz]), tol = tol,
		maxIter = maxiter)

	semi.cache.u[3,1:nx,1:nz] .= -reshape(u_new, (nx, nz))
	update_ghost_values!(cache, grid, boundary_conditions)
end

function compute_pressure!(semi, matrix_solver::BiCGSTABSolver)
	(; cache, grid, boundary_conditions) = semi
	(; div) = cache
	(;dx, dz, nx, nz) = grid
	update_ghost_values!(cache, grid, boundary_conditions)

	@unpack tol, maxiter = matrix_solver
	u_new, exit_code, num_iters = bicgstab(
		(x,u) -> laplace_2d!(x, u, nx, nz, dx, dz), vec(div[1:nx, 1:nz]), tol = tol,
		maxIter = maxiter)

	semi.cache.u[3,1:nx,1:nz] .= -reshape(u_new, (nx, nz))
	update_ghost_values!(cache, grid, boundary_conditions)
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

function compute_error(semi, t)
	(; cache, grid, equations, initial_condition) = semi
	(; u) = cache
	(; nx, nz, dx, dz) = grid
	(; xc, zc, xf, zf) = grid
	nvar = Val(nvariables(equations))

	l1, l2, linf = 0.0, 0.0, 0.0
	for i in 1:nx, k in 1:nz
		exact = initial_condition((xc[i], zc[k]), (xf[i], zf[k]), t, equations)
		u_node = get_node_vars(u, nvar, i, k)[1:2]
		error = abs(u_node[1] - exact[1]) + abs(u_node[2] - exact[2])
		l1 += sum(error) * dx * dz
		l2 += sum(error.^2) * dx * dz
		linf = max(linf, maximum(error))
	end
	l2 = sqrt(l2)
	return l1, l2, linf
end
