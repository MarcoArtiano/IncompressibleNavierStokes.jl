using Test
using TrixiBase
using IncompressibleNavierStokes
using IncompressibleNavierStokes: examples_dir

@testset "TGV test" begin
    tspan = (0.0, 12.0)
    nx = nz = 6
    trixi_include("$(examples_dir())/elixir_tgv_SOR.jl", tspan = tspan, nz = nz, nx = nx,
                  matrix_solver = SORSolver(maxiter = 1000, tol = 1e-12, om = 1.6))
    @test isapprox(sol.l1, 4.2136403258919313e-10, atol = 1e-10, rtol = 1e-10)
    @test isapprox(sol.l2, 2.554269898215263e-10, atol = 1e-10, rtol = 1e-10)
    @test isapprox(sol.linf, 2.4286713387346114e-10, atol = 1e-10, rtol = 1e-10)
end

@testset "TGV test" begin
    tspan = (0.0, 12.0)
    nx = nz = 6
    trixi_include("$(examples_dir())/elixir_tgv_BICGSTAB.jl", tspan = tspan, nz = nz, nx = nx,
                  matrix_solver = BiCGSTABSolver(maxiter = 1000, tol = 1e-12))
    @test isapprox(sol.l1, 6.941030663009133e-11, atol = 1e-10, rtol = 1e-10)
    @test isapprox(sol.l2, 4.117440312298273e-11, atol = 1e-10, rtol = 1e-10)
    @test isapprox(sol.linf, 5.405456589852789e-11, atol = 1e-10, rtol = 1e-10)
end

@testset "TGV test" begin
    tspan = (0.0, 12.0)
    nx = nz = 6
    trixi_include("$(examples_dir())/elixir_tgv_CG.jl", tspan = tspan, nz = nz, nx = nx,
                  matrix_solver = CGSolver(maxiter = 1000, tol = 1e-12))
    @test isapprox(sol.l1, 1.0898817476445089e-10, atol = 1e-10, rtol = 1e-10)
    @test isapprox(sol.l2, 4.117440312298273e-11, atol = 1e-10, rtol = 1e-10)
    @test isapprox(sol.linf, 5.405456589852789e-11, atol = 1e-10, rtol = 1e-10)
end
