using Trixi: AbstractEquations

"""
    AbstractSpatialSolver

Abstract type for solvers that specify the spatial discretization.
"""
abstract type AbstractSpatialSolver end

"""
    AbstractFiniteVolumeSolver

Finite volume spatial discretization.
"""
struct FiniteVolumeSolver <: AbstractSpatialSolver end

"""
    AbstractBoundaryCondition

Abstract type for boundary conditions.
"""
abstract type AbstractBoundaryCondition end

"""
    PeriodicBC

Specifies periodic boundary condition.
"""
struct PeriodicBC <: AbstractBoundaryCondition end

"""
    BoundaryConditions

Struct containing the left and right boundary conditions.
"""

struct BoundaryConditions{LeftBC, RightBC, BottomBC, TopBC}
    left::LeftBC
    right::RightBC
    bottom::BottomBC
    top::TopBC
    function BoundaryConditions(left, right, bottom, top)
       if left isa PeriodicBC || right isa PeriodicBC || bottom isa PeriodicBC || top isa PeriodicBC
          @assert left isa PeriodicBC && right isa PeriodicBC && bottom isa PeriodicBC && top isa PeriodicBC
       end
       new{typeof(left), typeof(right), typeof(bottom), typeof(top)}(left, right, bottom, top)
    end
end