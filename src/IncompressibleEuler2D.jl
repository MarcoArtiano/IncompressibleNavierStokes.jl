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
f_1 = (p + v_1^2, v_1*v_2)
f_2 = (v_1*v_2, p + v_2^2)
"""
function flux(u, orientation::Integer, equations::IncompressibleEuler2D)
    v1, v2, p = u
    # TODO: write the analytical divergence form flux:
    if orientation == 1
        return SVector(p + v1^2, v1*v2, zero(eltype(u)))
    else
        return SVector(v1*v2, p + v2^2, zero(eltype(u)))
    end
end

# Computes a flux approximation for the part of the flux that doesn't contain the pressure.
# Divergence form for staggered grid of the non-linear terms.
# This is equivalent to the advection form, since the divergence of the velocity field is 0.
function flux_div(u_ll, u_rr, u_dd, orientation::Integer, equations::IncompressibleEuler2D)
    v1_rr, v2_rr, p_rr = u_rr
    v1_ll, v2_ll, p_ll = u_ll
    v1_dd, v2_dd, p_dd = u_dd

    v1_avg = (v1_rr + v1_ll)*0.5f0
    v1_avg_vertical = (v1_rr + v1_dd)*0.5f0
    v2_avg = (v2_rr + v2_ll)*0.5f0
    v2_avg_vertical = (v2_rr + v2_dd)*0.5f0
    if orientation == 1
        f1 = v1_avg^2
        f2 = v1_avg_vertical*v2_avg
    else
        f1 = v1_avg*v2_avg_vertical
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
