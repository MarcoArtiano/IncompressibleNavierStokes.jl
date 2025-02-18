using IncompressibleNavierStokes

Nx = 10
Nz = 10
domain = (0.0, 1.0, 0.0, 2.0)
grid = mesh(domain, Nx, Nz)