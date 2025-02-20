using ConjugateGradientsGPU

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

function discrete_laplacian_2d!(y, x, Nx, Ny, dx, dy)
    function idx(x, y)
        return ((x - 1) % Nx + 1) + ((y - 1) % Ny) * Nx
    end

    for i in 1:Nx, j in 1:Ny
        I = idx(i, j)
        im1 = (i == 1)  ? Nx : i - 1  # Wrap around for periodic BC
		ip1 = (i == Nx) ? 1  : i + 1  # Wrap around for periodic BC

        jm1 = (j == 1)  ? Ny : j - 1  # Wrap around for periodic BC
		jp1 = (j == Ny) ? 1  : j + 1  # Wrap around for periodic BC
        y[I] += -4.0 * x[I] / dx^2 # L[I, I] = -4 (TODO - Fix dy)
        y[I] += x[idx(ip1, j)] / dx^2 # L[I, idx(ip1, j)] = 1
        y[I] += x[idx(im1, j)] / dx^2 # L[I, idx(im1, j)] = 1
        y[I] += x[idx(i, jp1)] / dy^2 # L[I, idx(i, jp1)] = 1
        y[I] += x[idx(i, jm1)] / dy^2 # L[I, idx(i, jm1)] = 1
    end

    return y
end

nx = nz = 16
rhs = ones(nx, nz)

dx = 1.0 / nx
dz = 1.0 / nz

function laplace_2d_vec(x::Vector{T}, nx, nz, dx, dz) where T
    u = reshape(x, nx, nz)  # Reshape x into a matrix
    x_out = similar(u)      # Output storage
    laplace_2d!(x_out, u, nx, nz, dx, dz)  # Apply Laplacian
    return vec(x_out)       # Return as vector for AD
end

# Create input vector
x0 = vec(rhs)  # Flatten to 1D

# Compute Jacobian
J = ForwardDiff.jacobian(x -> laplace_2d_vec(x, nx, nz, dx, dz), x0)

println(size(J))  # Should be (nx*nz, nx*nz)

for i = 1:nx
    for k = 1:nz
        rhs[i, k] = sin(2π*grid_x[i]) * sin(2π*grid_x[k])
    end
end


rhs .-= avg(rhs)
u_new, exit_code, num_iters = bicgstab((y,x) ->discrete_laplacian_2d!(y, x, nx, nz, dx, dz),
    # (x,u) -> laplace_2d!(x, u, nx, nz, dx, dz),
                                 vec(rhs))

if exit_code == -13
    println("CG failed terribly")
end

u_new

using ConjugateGradientsGPU

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




###

using ConjugateGradientsGPU

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

            # Apply Laplacian in both x and z directions
            lap_x = (u_mat[ip1, k] - 2.0f0 * u_mat[i, k] + u_mat[im1, k]) / dx^2
            lap_z = (u_mat[i, kp1] - 2.0f0 * u_mat[i, k] + u_mat[i, km1]) / dz^2

            x_mat[i, k] = lap_x + lap_z  # Combine results for both directions
        end
    end

    x .*= -1.0
end

nx = nz = 10
rhs = ones(nx, nz)

dx = 1.0 / nx
dz = 1.0 / nz

function laplace_2d_vec(x::Vector{T}, nx, nz, dx, dz) where T
    u = reshape(x, nx, nz)  # Reshape x into a matrix
    x_out = similar(u)      # Output storage
    laplace_2d!(x_out, u, nx, nz, dx, dz)  # Apply Laplacian
    return vec(x_out)       # Return as vector for AD
end

# Create input vector
x0 = vec(rhs)  # Flatten to 1D

# Compute Jacobian
J = ForwardDiff.jacobian(x -> laplace_2d_vec(x, nx, nz, dx, dz), x0)

println(size(J))  # Should be (nx*nz, nx*nz)

J

using Enzyme

function laplace_2d_enzyme(x, nx, nz, dx, dz)
    u = reshape(x, nx, nz)
    x_out = similar(u)
    laplace_2d!(x_out, u, nx, nz, dx, dz)
    return vec(x_out)
end

# Initialize Jacobian storage
J = zeros(length(x0), length(x0))

# Compute Jacobian using Enzyme
a, = Enzyme.autodiff(Enzyme.Forward, laplace_2d_enzyme, Duplicated(ones(nx * nx), ones(nx * nx)), Const(nx), Const(nz), Const(dx), Const(dz))

println(size(J))  # Should be (nx*nz, nx*nz)

a

function discrete_laplacian_2d(Nx, Ny)
    N = Nx * Ny
    L = zeros(N, N)

    dx = 1.0 / Nx^2

    function idx(x, y)
        return ((x - 1) % Nx + 1) + ((y - 1) % Ny) * Nx
    end

    for i in 1:Nx, j in 1:Ny
        I = idx(i, j)
        im1 = (i == 1)  ? Nx : i - 1  # Wrap around for periodic BC
		ip1 = (i == Nx) ? 1  : i + 1  # Wrap around for periodic BC

        jm1 = (j == 1)  ? Ny : j - 1  # Wrap around for periodic BC
		jp1 = (j == Ny) ? 1  : j + 1  # Wrap around for periodic BC
        L[I, I] = -4
        L[I, idx(ip1, j)] = 1
        L[I, idx(im1, j)] = 1
        L[I, idx(i, jp1)] = 1
        L[I, idx(i, jm1)] = 1
    end

    return L ./ dx^2
end

function discrete_laplacian_2d!(y, x, Nx, Ny)
    dx = 1.0 / Nx^2
    function idx(x, y)
        return ((x - 1) % Nx + 1) + ((y - 1) % Ny) * Nx
    end

    for i in 1:Nx, j in 1:Ny
        I = idx(i, j)
        im1 = (i == 1)  ? Nx : i - 1  # Wrap around for periodic BC
		ip1 = (i == Nx) ? 1  : i + 1  # Wrap around for periodic BC

        jm1 = (j == 1)  ? Ny : j - 1  # Wrap around for periodic BC
		jp1 = (j == Ny) ? 1  : j + 1  # Wrap around for periodic BC
        y[I] += -4.0 * x[I] / dx^2 # L[I, I] = -4
        y[I] += x[idx(ip1, j)] / dx^2 # L[I, idx(ip1, j)] = 1
        y[I] += x[idx(im1, j)] / dx^2 # L[I, idx(im1, j)] = 1
        y[I] += x[idx(i, jp1)] / dx^2 # L[I, idx(i, jp1)] = 1
        y[I] += x[idx(i, jm1)] / dx^2 # L[I, idx(i, jm1)] = 1
    end

    return y
end

J = discrete_laplacian_2d(8, 8)


x = rand(64)
J*x

y = zeros(64)
discrete_laplacian_2d!(y, x, 8, 8)

@assert isapprox(J*x, y) J*x .- y

J*x .- y