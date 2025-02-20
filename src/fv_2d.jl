import Trixi: get_node_vars
using ConjugateGradientsGPU

function update_solution!(semi, dt)
	(; cache, equations, boundary_conditions, grid) = semi
	(; u, du, backend) = cache
	@. du = 0.0f0
	update_ghost_values!(equations, cache, grid, boundary_conditions, backend)
	compute_surface_fluxes!(semi, backend) # Approximations of flux without pressure
	update_rhs!(semi, backend) # Compute RHS without pressure term
	@. u.parent -= dt * du.parent # Evolve solution without pressure term
    apply_correction!(semi) # Include pressure term correction

end

function update_ghost_values!(equations, cache, grid::CartesianGrid2D,
							  boundary_conditions::BoundaryConditions,
							  backend)

	apply_left_bc!(equations, cache, boundary_conditions.left, grid.nx, backend)
	apply_right_bc!(equations, cache, boundary_conditions.right, grid.nx, backend)
	apply_bottom_bc!(equations, cache, boundary_conditions.bottom, grid.nz, backend)
	apply_top_bc!(equations, cache, boundary_conditions.top, grid.nz, backend)

end

function apply_right_bc!(equations, cache, right::PeriodicBC, nx, backend::MyCPU)
	(; u) = cache
	u[:, nx+1, :] .= @views u[:, 1, :]
end

function apply_left_bc!(equations, cache, left::PeriodicBC, nx, backend::MyCPU)
	(; u) = cache
	u[:, 0, :] .= @views u[:, nx, :]
end

function apply_bottom_bc!(equations, cache, bottom::PeriodicBC, nz, backend::MyCPU)
	(; u) = cache
	u[:, :, 0] .= @views u[:, :, nz]
end

function apply_top_bc!(equations, cache, top::PeriodicBC, nz, backend::MyCPU)
	(; u) = cache
	u[:, :, nz+1] .= @views u[:, :, 1]
end

function apply_right_bc!(equations, cache, right::PeriodicBC, nx, backend::Union{CPU, GPU})
	(; u, backend) = cache
	nvar = Val(nvariables(equations))
	apply_right_bc_periodic_kernel!(backend)(u, nvar, nx, ndrange = nx)
end

@kernel function apply_right_bc_periodic_kernel!(u, nvar::Val{N}, nx) where {N}
	k = @index(Global, Linear)
	for n = 1:N
		u[n, nx+1, k] = u[n, 1, k]
	end
end

function apply_left_bc!(equations, cache, left::PeriodicBC, nx, backend::Union{CPU, GPU})
	(; u, backend) = cache
	nvar = Val(nvariables(equations))
	apply_left_bc_periodic_kernel!(backend)(u, nvar, nx, ndrange = nx)
end

@kernel function apply_left_bc_periodic_kernel!(u, nvar::Val{N}, nx) where {N}
	k = @index(Global, Linear)
	for n = 1:N
		u[n, 0, k] = u[n, nx, k]
	end
end

function apply_bottom_bc!(equations, cache, bottom::PeriodicBC, nz, backend::Union{CPU, GPU})
	(; u, backend) = cache
	nvar = Val(nvariables(equations))
	apply_bottom_bc_periodic_kernel!(backend)(u, nvar, nz, ndrange = nz)
end

@kernel function apply_bottom_bc_periodic_kernel!(u, nvar::Val{N}, nz) where {N}
	i = @index(Global, Linear)
	for n = 1:N
		u[n, i, 0] = u[n, i, nz]
	end
end

function apply_top_bc!(equations, cache, top::PeriodicBC, nz, backend::Union{CPU, GPU})
	(; u, backend) = cache
	nvar = Val(nvariables(equations))
	apply_top_bc_periodic_kernel!(backend)(u, nvar, nz, ndrange = nz)
end

@kernel function apply_top_bc_periodic_kernel!(u, nvar::Val{N}, nz) where {N}
	i = @index(Global, Linear)
	for n = 1:N
		u[n, i, nz+1] = u[n, i, 1]
	end
end

@inline function get_node_vars(u, ::Val{N}, indices...) where {N}
    SVector(ntuple(@inline(v->u[v, indices...]), N))
end

@inline function set_node_vars!(u, v, ::Val{N}, indices...) where {N}
	for i in 1:N
		u[i, indices...] = v[i]
	end
end


function compute_surface_fluxes!(semi, backend::MyCPU)
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

function compute_surface_fluxes!(semi, backend::Union{CPU, GPU})
	(; grid, equations, surface_flux, cache) = semi
	(; nx, nz) = grid
	(; u, fu) = cache

	nvar = Val(nvariables(equations))
	compute_surface_flux_kernel(backend)(u, fu, surface_flux, nvar, equations,
										 ndrange = (nx+1, nz+1))
end

@kernel function compute_surface_flux_kernel(u, fu, surface_flux, nvar, equations)
	i, k = @index(Global, NTuple)

	u_rr = get_node_vars(u, nvar, i, k)
	u_ll = get_node_vars(u, nvar, i-1, k)
	u_dd = get_node_vars(u, nvar, i, k-1)

	flux_x = surface_flux(u_ll, u_rr, u_dd, 1, equations)
	flux_z = surface_flux(u_dd, u_rr, u_ll, 2, equations)
	set_node_vars!(fu, flux_x, nvar, i, k, 1)
	set_node_vars!(fu, flux_z, nvar, i, k, 2)
end

function update_rhs!(semi, backend::MyCPU)
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

function update_rhs!(semi, backend::Union{CPU, GPU})
	(; cache, grid, equations) = semi

	(; nx, nz, dx, dz) = grid
	(; fu, du) = cache
	nvar = Val(nvariables(equations))
	update_rhs_kernel!(backend)(fu, du, dx, dz, nvar, ndrange = (nx, nz))

end

@kernel function update_rhs_kernel!(fu, du, dx, dz, nvar)
	i, k = @index(Global, NTuple)

	orientation = 1
	fn_rr = get_node_vars(fu, nvar, i+1, k, orientation)
	fn_ll = get_node_vars(fu, nvar, i, k, orientation)

	orientation = 2
	gn_rr = get_node_vars(fu, nvar, i, k+1, orientation)
	gn_ll = get_node_vars(fu, nvar, i, k, orientation)
	rhs = (fn_rr - fn_ll) / dx + (gn_rr - gn_ll) / dz
	set_node_vars!(du, rhs, nvar, i, k)
end

# TODO: to add a struct for Poisson solver: div flux, p flux, poisson parameters

function apply_correction!(semi)
	(; backend) = semi.cache
    compute_div!(semi, backend) # Divergence of the velocity field evolved without pressure term
    compute_pressure!(semi, semi.matrix_solver) # Solve for pressure as Δp = div(u)
    project_pressure!(semi, backend) # Include the pressure term derivative in the evolution

end

function compute_div!(semi, backend::MyCPU)

    (; cache, grid) = semi
    (; u, div) = cache
    (; nx,nz, dx, dz) = grid

    for i = 1:nx
        for k = 1:nz
            div[i,k] = (u[1,i+1,k] - u[1,i,k])/dx + (u[2,i,k+1] - u[2,i,k])/dz
        end
    end

end

function compute_div!(semi, backend::Union{CPU, GPU})

    (; cache, grid) = semi
    (; u, div) = cache
    (; nx,nz, dx, dz) = grid

    compute_div_kernel(backend)(u, div, dx, dz, ndrange = (nx, nz))

end

@kernel function compute_div_kernel(u, div, dx, dz)
	i, k = @index(Global, NTuple)
	div[i, k] = (u[1, i+1, k] - u[1, i, k]) / dx + (u[2, i, k+1] - u[2, i, k]) / dz
end

function compute_pressure!(semi, matrix_solver::SORSolver)
    # TODO: Move into a struct:
    @unpack tol, maxiter, om = matrix_solver
    normres = 1
    (; cache, grid, boundary_conditions, equations) = semi
    (; u, div, normatrix, backend) = cache
    (; dx, dz, nx, nz) = grid
    #add max iter
	it = 0
    while normres > tol
		if it > maxiter
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
	    update_ghost_values!(equations, cache, grid, boundary_conditions, backend)

        # This is not efficient
        for i = 1:nx
            for k = 1:nz
        normatrix[i,k] = abs((u[3,i+1,k] -2*u[3,i,k]+ u[3,i-1,k])/dx^2 + (u[3,i,k-1] + u[3,i,k+1] - 2*u[3,i,k])/dz^2 -div[i,k])
            end
        end
        normres = maximum(normatrix)
    end
end

function laplace_2d!(x, u, nx, nz, dx, dz, backend::MyCPU)
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

	# Solve -Δ because that is positive definite
	x .*= -1.0
end

function laplace_2d!(x, u, nx, nz, dx, dz, backend::Union{CPU, GPU})
	x_mat = reshape(x, nx, nz)
	u_mat = reshape(u, nx, nz)

	laplace_2d_kernel!(backend)(x_mat, u_mat, dx, dz, nx, nz, ndrange = (nx, nz))
end

@kernel function laplace_2d_kernel!(x, u, dx, dz, nx, nz)
	i, k = @index(Global, NTuple)

	# TODO - These `if` conditions do not sound GPU efficient
	im1 = (i == 1)  ? nx : i - 1  # Wrap around for periodic BC
	ip1 = (i == nx) ? 1  : i + 1  # Wrap around for periodic BC

	km1 = (k == 1)  ? nz : k - 1  # Wrap around for periodic BC
	kp1 = (k == nz) ? 1  : k + 1  # Wrap around for periodic BC

	x[i, k]  = (u[ip1, k] - 2.0f0 * u[i, k] + u[im1, k]) / dx^2  # x direction
	x[i, k] += (u[i, km1] - 2.0f0 * u[i, k] + u[i, kp1]) / dz^2  # z direction
	x[i, k] *= -1.0
end

function compute_pressure!(semi, matrix_solver::CGSolver)
	(; cache, grid, equations, boundary_conditions) = semi
	(; div, backend) = cache
	(;dx, dz, nx, nz) = grid
	update_ghost_values!(equations, cache, grid, boundary_conditions, backend)

	@unpack tol, maxiter = matrix_solver
	u_new, exit_code, num_iters = cg(
		(x,u) -> laplace_2d!(x, u, nx, nz, dx, dz, backend),
							 vec(div[1:nx, 1:nz]), tol = tol,
							 maxIter = maxiter)

	# Since -Δ was solved (for positive definiteness) and physical equation has +Δ
	semi.cache.u[3,1:nx,1:nz] .= -reshape(u_new, (nx, nz))
	update_ghost_values!(equations, cache, grid, boundary_conditions, backend)
end

function compute_pressure!(semi, matrix_solver::BiCGSTABSolver)
	(; cache, grid, boundary_conditions, equations) = semi
	(; div, backend) = cache
	(; dx, dz, nx, nz) = grid
	update_ghost_values!(equations, cache, grid, boundary_conditions, backend)

	@unpack tol, maxiter = matrix_solver
	u_new, exit_code, num_iters = bicgstab(
		(x,u) -> laplace_2d!(x, u, nx, nz, dx, dz, backend),
							 vec(div[1:nx, 1:nz]), tol = tol,
							 maxIter = maxiter)

	# Since -Δ was solved (for positive definiteness) and physical equation has +Δ
	semi.cache.u[3,1:nx,1:nz] .= -reshape(u_new, (nx, nz))
	update_ghost_values!(equations, cache, grid, boundary_conditions, backend)
end

function project_pressure!(semi, backend::MyCPU)

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

function project_pressure!(semi, backend::Union{CPU, GPU})

    (; cache, grid) = semi
    (; u) = cache
    (; dx, dz, nx, nz) = grid

	project_pressure_kernel(backend)(u, dx, dz, ndrange = (nx, nz))
end

@kernel function project_pressure_kernel(u, dx, dz)
	i, k = @index(Global, NTuple)
	u[1, i, k] = u[1, i, k] - (u[3, i, k] - u[3, i-1, k]) / dx
	u[2, i, k] = u[2, i, k] - (u[3, i, k] - u[3, i, k-1]) / dz
end

function compute_error(semi, t, backend::MyCPU)
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

function compute_error(semi, t, backend::Union{CPU, GPU})
	(; cache, grid, equations, initial_condition) = semi
	(; u) = cache
	(; nx, nz, dx, dz) = grid
	(; xc, zc, xf, zf) = grid
	nvar = Val(nvariables(equations))

	compute_error_kernel!(backend)(cache.error_array, initial_condition,
								   u, xc, zc, xf, zf, t,
								   equations, ndrange = (nx, nz))

	l1 = sum(cache.error_array) * dx * dz
	l2 = sqrt(sum(cache.error_array.^2) * dx * dz)
	linf = maximum(cache.error_array)
	return l1, l2, linf
end

@kernel function compute_error_kernel!(error_array,
									   initial_condition, u, xc, zc, xf, zf, t, equations)
	i, k = @index(Global, NTuple)
	nvar = Val(nvariables(equations))
	exact = initial_condition((xc[i], zc[k]), (xf[i], zf[k]), t, equations)
	u_node = get_node_vars(u, nvar, i, k)
	error = abs(u_node[1] - exact[1]) + abs(u_node[2] - exact[2])
	error_array[i, k] = error
end
