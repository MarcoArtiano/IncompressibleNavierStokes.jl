using IncompressibleNavierStokes

nx = 10
nz = 10
domain = (0.0, 1.0, 0.0, 2.0)
grid = mesh(domain, nx, nz)

semi = SemiDiscretization(grid, nothing, nothing, nothing)

