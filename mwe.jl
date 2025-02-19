using ConjugateGradients

function avg(u)
    sum(u) / length(u)
end

function laplace_2d!(x, u, nx, nz, dx, dz)
    u .-= avg(u)

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
end

nx = nz = 10
rhs = ones(nx, nz)

for i = 1:nx
    for k = 1:nz
        rhs[i, k] = sin(2π*grid_x[i]) * sin(2π*grid_x[k])
    end
end

dx = 1.0 / nx
dz = 1.0 / nz

rhs .-= avg(rhs)
u_new, exit_code, num_iters = bicgstab((x,u) -> laplace_2d!(x, u, nx, nz, dx, dz),
                                 vec(rhs))

if exit_code == -13
    println("CG failed terribly")
end

u_new

using ConjugateGradients

function u_minus_laplacian_2d!(x, u, nx, nz, dx, dz)
	x_mat = reshape(x, nx, nz)
	u_mat = reshape(u, nx, nz)

	for i = 1:nx
		im1 = (i == 1)  ? nx : i - 1  # Wrap around for periodic BC
		ip1 = (i == nx) ? 1  : i + 1  # Wrap around for periodic BC

		for k = 1:nz
			km1 = (k == 1)  ? nz : k - 1  # Wrap around for periodic BC
			kp1 = (k == nz) ? 1  : k + 1  # Wrap around for periodic BC

			x_mat[i, k]  = -(u_mat[ip1, k] - 2.0f0 * u_mat[i, k] + u_mat[im1, k]) / dx^2  # x direction
			x_mat[i, k] += -(u_mat[i, km1] - 2.0f0 * u_mat[i, k] + u_mat[i, kp1]) / dz^2  # z direction

            x_mat[i, k] += u_mat[i, k]
		end
	end
end

nx = nz = 100

rhs = zeros(nx, nz)

grid_x = LinRange(0, 1, nx) # grid

for i = 1:nx
    for k = 1:nz
        rhs[i, k] = sin(2π*grid_x[i]) * sin(2π*grid_x[k])
    end
end

dx = 1.0 / nx
dz = 1.0 / nz

u_new, exit_code, num_iters = cg((x,u) -> u_minus_laplacian_2d!(x, u, nx, nz, dx, dz),
                                 vec(rhs))

# Compute exact solution (Fourier mode solution)
exact_solution = zeros(nx, nz)
λ = 1 + (2π)^2 + (2π)^2
for i = 1:nx
    for k = 1:nz
        exact_solution[i, k] = rhs[i, k] / λ
    end
end

# Compute error norms
nu_sol_mat = reshape(u_new, nx, nz)
error = abs.(nu_sol_mat - exact_solution)
L2_error = sqrt(sum(error .^ 2) / (nx * nz))
max_error = maximum(error)

println("L2 error: ", L2_error)
println("Max error: ", max_error)

