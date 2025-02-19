using Test
using TrixiBase
using IncompressibleNavierStokes
using IncompressibleNavierStokes: examples_dir

@testset "TGV test" begin
    tspan = (0.0, 12.0)
    nx = nz = 6
    trixi_include("$(examples_dir())/elixir_tgv.jl", tspan = tspan, nz = nz, nx = nx)
    @test isapprox(sol.l1, 4.2136403258919313e-10, atol = 1e-10, rtol = 1e-10)
    @test isapprox(sol.l2, 2.554269898215263e-10, atol = 1e-10, rtol = 1e-10)
    @test isapprox(sol.linf, 2.4286713387346114e-10, atol = 1e-10, rtol = 1e-10)
end
