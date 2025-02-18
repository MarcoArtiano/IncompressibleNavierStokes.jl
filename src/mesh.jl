
struct CartesianGrid2D{RealT <: Real, ArrayType1, ArrayType2}
    domain::Tuple{RealT,RealT,RealT,RealT}  # xmin, xmax, zmin, zmax
    nx::Int             # Nx - number of points in the horizontal direction        
    nz::Int             # Nz - number of points in the vertical direction
    xc::ArrayType1      # Cell centers of the elements in the horizontal direction
    zc::ArrayType2      # Cell centers of the elements in the vertical direction
    xf::ArrayType1      # Cell faces of the elements in the horizontal direction
    zf::ArrayType2      # Cell faces of the elements in the vertical direction
    dx::RealT           
    dz::RealT
    nbx::Int
    nbz::Int
end

# TODO: make Nx and Nz a tuple

function mesh(domain::Tuple{<:Real, <:Real, <:Real, <:Real}, nx, nz; nbx = 1, nbz = 1)

    xmin, xmax, zmin, zmax = domain

    RealT = eltype(domain)
    @assert xmin < xmax
    @assert zmin < zmax

    dx = (xmax - xmin)/nx
    dz = (zmax - zmin)/nz

    ## The names are wrong:
    # For Harlow-Welch or C-Grid Arakawa the nodes are where the pressure is stored
    # while the center cells are the center of the cells defined by the pressure nodes
    # the velocities instead are staggered.
    # Nbx and Nbz are the number of ghost cells in the x and z direction
    # For Nbx = 0 and Nbz = 0 this is broken!!
    ## Creating the cell centers
    xc_ = LinRange(xmin - 0.5f0*dx -(nbx-1)*dx, xmax - 0.5f0*dx +(nbx-1)*dx, (nx-1) + 2*nbx)
    zc_ = LinRange(zmin - 0.5f0*dz -(nbz-1)*dz, zmax - 0.5f0*dz +(nbz-1)*dz, (nz-1) + 2*nbz)
    xc = OffsetArray(xc_, OffsetArrays.Origin(1-nbx))
    zc = OffsetArray(zc_, OffsetArrays.Origin(1-nbz))
    
    ## Creating the cell faces
    xf_ = LinRange(xmin - nbx*dx, xmax + (nbx-1)*dx, nx + 2*nbx)
    zf_ = LinRange(zmin - nbz*dz, zmax + (nbz-1)*dz, nz + 2*nbz)
    xf = OffsetArray(xf_, OffsetArrays.Origin(1-nbx))
    zf = OffsetArray(zf_, OffsetArrays.Origin(1-nbz))

    return CartesianGrid2D(domain, nx, nz, xc, zc, xf, zf, dx, dz, nbx, nbz)

end
