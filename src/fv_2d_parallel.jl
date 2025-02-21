## Add this to keep track of the order of the operations
#function update_solution!(semi, dt)
#	(; cache, equations, boundary_conditions, grid) = semi
#	(; u, du, backend) = cache
#	@. du = 0.0f0
#	update_ghost_values!(equations, cache, grid, boundary_conditions, backend)
#	compute_surface_fluxes!(semi, backend) # Approximations of flux without pressure
#	update_rhs!(semi, backend) # Compute RHS without pressure term
#	@. u.parent -= dt * du.parent # Evolve solution without pressure term
#    apply_correction!(semi) # Include pressure term correction

#end

# TODO: to add a struct for Poisson solver: div flux, p flux, poisson parameters

#function apply_correction!(semi)
#	(; backend) = semi.cache
#    compute_div!(semi, backend) # Divergence of the velocity field evolved without pressure term
#    compute_pressure!(semi, semi.matrix_solver) # Solve for pressure as Δp = div(u)
#    project_pressure!(semi, backend) # Include the pressure term derivative in the evolution

#end


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

@inline function set_node_vars!(u, v, ::Val{N}, indices...) where {N}
	for i in 1:N
		u[i, indices...] = v[i]
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

# this is likely slow because of "overlapped readings"

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

function update_rhs!(semi, backend::Union{CPU, GPU})
	(; cache, grid, equations) = semi

	(; nx, nz, dx, dz) = grid
	(; fu, du) = cache
	nvar = Val(nvariables(equations))
	update_rhs_kernel!(backend)(fu, du, dx, dz, nvar, ndrange = (nx, nz))

end

# this one might be fine

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

function compute_div!(semi, backend::Union{CPU, GPU})

    (; cache, grid) = semi
    (; u, div) = cache
    (; nx,nz, dx, dz) = grid

    compute_div_kernel(backend)(u, div, dx, dz, ndrange = (nx, nz))

end

# this might be also fine

@kernel function compute_div_kernel(u, div, dx, dz)
	i, k = @index(Global, NTuple)
	div[i, k] = (u[1, i+1, k] - u[1, i, k]) / dx + (u[2, i, k+1] - u[2, i, k]) / dz
end

function laplace_2d!(x, u, nx, nz, dx, dz, backend::Union{CPU, GPU})
	x_mat = reshape(x, nx, nz)
	u_mat = reshape(u, nx, nz)

	laplace_2d_kernel!(backend)(x_mat, u_mat, dx, dz, nx, nz, ndrange = (nx, nz))
end

# this might not be fine because of overlapping readings

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

function project_pressure!(semi, backend::Union{CPU, GPU})

    (; cache, grid) = semi
    (; u) = cache
    (; dx, dz, nx, nz) = grid

	project_pressure_kernel(backend)(u, dx, dz, ndrange = (nx, nz))
end

# this might be fine

@kernel function project_pressure_kernel(u, dx, dz)
	i, k = @index(Global, NTuple)
	u[1, i, k] = u[1, i, k] - (u[3, i, k] - u[3, i-1, k]) / dx
	u[2, i, k] = u[2, i, k] - (u[3, i, k] - u[3, i, k-1]) / dz
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

# this also looks fine

@kernel function compute_error_kernel!(error_array,
									   initial_condition, u, xc, zc, xf, zf, t, equations)
	i, k = @index(Global, NTuple)
	nvar = Val(nvariables(equations))
	exact = initial_condition((xc[i], zc[k]), (xf[i], zf[k]), t, equations)
	u_node = get_node_vars(u, nvar, i, k)
	error = abs(u_node[1] - exact[1]) + abs(u_node[2] - exact[2])
	error_array[i, k] = error
end