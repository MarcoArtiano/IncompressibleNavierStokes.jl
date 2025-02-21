"""
    ODE

Struct containing semidiscretization plus the time interval.
"""
struct ODE{Semi <: SemiDiscretization, RealT <: Real}
    semi::Semi
    tspan::Tuple{RealT, RealT}
end

function solve(ode::ODE, dt; maxiters = nothing, analysis_interval = 1000)
    (; semi, tspan) = ode
    (; cache) = semi
    (; backend) = cache
    Tf = tspan[2]

    it, t = 0, 0.0f0
    while t < Tf
        if t + dt > Tf
            dt = Tf - t
        end

       update_solution!(semi, dt)
       l1, l2, linf = compute_error(semi, t, backend)

       t += dt; it += 1

       if it % analysis_interval == 0
        @show t, dt, it
        @show l1, l2, linf
       end
    end

    l1, l2, linf = compute_error(semi, t, backend)

    sol = (; cache.u, semi,
           l1, l2, linf
           )
    return sol
end