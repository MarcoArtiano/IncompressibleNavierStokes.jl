using Test
using TrixiBase
using IncompressibleNavierStokes
using IncompressibleNavierStokes: examples_dir

@testset "TGV test" begin
    tspan = (0.0, 12.0)
    nx = nz = 6
    trixi_include("$(examples_dir())/elixir_tgv_SOR.jl", tspan = tspan, nz = nz, nx = nx,
                  matrix_solver = SORSolver(maxiter = 1000, tol = 1e-14, om = 1.6))
    @test isapprox(sol.l1, 1.5146884507158896e-10, atol = 1e-9, rtol = 1e-9)
    @test isapprox(sol.l2, 9.090646452392059e-11, atol = 1e-9, rtol = 1e-9)
    @test isapprox(sol.linf, 9.000963516381447e-11, atol = 1e-9, rtol = 1e-9)
end

@testset "TGV test" begin
    tspan = (0.0, 12.0)
    nx = nz = 6
    trixi_include("$(examples_dir())/elixir_tgv_BICGSTAB.jl", tspan = tspan, nz = nz, nx = nx,
                  matrix_solver = BiCGSTABSolver(maxiter = 1000, tol = 1e-12))
    @test isapprox(sol.l1, 2.657132595352477e-10, atol = 1e-9, rtol = 1e-9)
    @test isapprox(sol.l2, 1.5975251750362167e-10, atol = 1e-9, rtol = 1e-9)
    @test isapprox(sol.linf, 1.548059931957685e-10, atol = 1e-9, rtol = 1e-9)
end

@testset "TGV test" begin
    tspan = (0.0, 12.0)
    nx = nz = 6
    trixi_include("$(examples_dir())/elixir_tgv_CG.jl", tspan = tspan, nz = nz, nx = nx,
                  matrix_solver = CGSolver(maxiter = 1000, tol = 1e-12))
    @test isapprox(sol.l1, 1.0898817476445089e-10, atol = 1e-9, rtol = 1e-9)
    @test isapprox(sol.l2, 4.117440312298273e-11, atol = 1e-9, rtol = 1e-9)
    @test isapprox(sol.linf, 5.405456589852789e-11, atol = 1e-9, rtol = 1e-9)
end


@testset "TGV test CG Float32" begin
    RealT = Float32
    tspan = map(RealT, (0.0, 12.0))
    tol = map(RealT, 0.5e-5)
    nx = nz = 16
    trixi_include("$(examples_dir())/elixir_tgv_CG.jl", tspan = tspan, nz = nz, nx = nx,
                  RealT = RealT,
                  matrix_solver = CGSolver(maxiter = 1000, tol = tol))
    @test isapprox(sol.l1, 7.953803598513033e-7, atol = 1e-4, rtol = 1e-6)
    @test isapprox(sol.l2, 1.2340538743632972e-6, atol = 1e-4, rtol = 1e-6)
    @test isapprox(sol.linf, 1.1376580005162396e-5, atol = 1e-4, rtol = 1e-6)
end