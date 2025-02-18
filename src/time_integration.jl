"""
    ODE

Struct containing semidiscretization plus the time interval.
"""
struct ODE{Semi <: SemiDiscretization, RealT <: Real}
    semi::Semi
    tspan::Tuple{RealT, RealT}
end

function solve(ode::ODE, dt; maxiters = nothing)
    (; semi, tspan) = ode
    (; cache) = semi
    Tf = tspan[2]

    it, t = 0, 0.0f0
    while t < Tf
        if t + dt > Tf
            dt = Tf - t
        end

       update_solution!(semi, dt)

       t += dt; it += 1
       @show t, dt, it
    end

    sol = (; cache.u, semi)
    return sol
end