using Test
using TrixiBase
using IncompressibleNavierStokes
using IncompressibleNavierStokes: examples_dir

@testset "TGV test" begin
    tspan = (0.0, 0.001)
    nx = nz = 10
    trixi_include("$(examples_dir())/elixir_tgv.jl", tspan = tspan, nz = nz, nx = nx)
    @test isapprox(sol.l1, 0.820865529154228, atol = 1e-9, rtol = 1e-9)
    @test isapprox(sol.l2, 0.5020826581461258, atol = 1e-9, rtol = 1e-9)
    @test isapprox(sol.linf, 0.5106697940787855, atol = 1e-9, rtol = 1e-9)
end
