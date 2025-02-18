import Trixi: varnames, prim2cons, cons2prim, cons2cons, max_abs_speeds, flux
using Trixi: nvariables, AbstractEquations

"""
    IncompressibleEuler2D

2D Incompressible Euler equations.
"""
struct IncompressibleEuler2D{RealT <: Real} <: AbstractEquations{2, 3}
    gamma::RealT
    rho::RealT
end

"""
    flux(u, orientation::Integer, equations::IncompressibleEuler2D)

Compute the flux for the IncompressibleEuler2D equations.
"""
function flux(u, orientation::Integer, equations::IncompressibleEuler2D)
    v1, v2, p = u
    
    return SVector(v1, v2, zero(eltype(u)))
end

function flux_test(u_ll, u_rr, orientation::Integer, equations::IncompressibleEuler2D)
    v1_ll, v2_ll = u_ll
    v1_rr, v2_rr = u_rr
    v1_avg = (v1_ll + v1_rr)/2
    v2_avg = (v2_ll + v2_rr)/2
    if orientation == 1
    f1 = v1_avg^2
    f2 = v1_avg*v2_avg
    else
    f1 = v1_avg*v2_avg
    f2 = v2_avg^2
    end
    return SVector(f1, f2, zero(eltype(u_ll)))
end

"""
    varnames(::IncompressibleEuler2D)

Return the variable names for the IncompressibleEuler2D equations.
"""
function varnames(::typeof(cons2cons), ::IncompressibleEuler2D)
    return ("u", "w", "p")
end

"""
    prim2cons(u, equations::IncompressibleEuler2D)

Convert primitive to conservative variables for the IncompressibleEuler2D equations.
"""
function prim2cons(u, equations::IncompressibleEuler2D)
    return u
end

"""
    cons2prim(u, equations::IncompressibleEuler2D)

Convert conservative to primitive variables for the IncompressibleEuler2D equations.
"""
function cons2prim(u, equations::IncompressibleEuler2D)
    return u
end

"""
    cons2cons(u, equations::IncompressibleEuler2D)

Convert conservative to conservative variables for the Euler1D equations.
"""
function cons2cons(u, equations::IncompressibleEuler2D)
    return u
end

"""
    max_abs_speeds(u, equation::IncompressibleEuler2D)

Compute the maximum absolute wave speeds for the IncompressibleEuler2D equations, which are
the eigen values of the flux.
"""
@inline function max_abs_speeds(u, equations::IncompressibleEuler2D)
    v1, v2 = u
    (; gamma, rho) = equations
    c = sqrt(gamma*p/rho) # sound speed
    return abs(v1) + c, abs(v2) + c
end