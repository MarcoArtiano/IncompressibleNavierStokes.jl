 #contourf(grid.xf[1:grid.nx], grid.zc[1:grid.nz], sol[1,1:grid.nx,1:grid.nz])

 throw(error)
 fig1 = Figure()
 ax = Axis(fig1[1,1], title="pressure 1")
 
 #hm = contourf!(ax,grid.xf[1:grid.nx], grid.zc[1:grid.nz], semi.cache.div[1:grid.nx,1:grid.nz])
 hm = contourf!(ax,grid.xf[1:grid.nx], grid.zf[1:grid.nz], semi.cache.u[3,1:grid.nx,1:grid.nz])

 Colorbar(fig1[1,2], hm)
 display(fig1)
 @show semi.cache.u[3,:,:]

 @show semi.cache.u[3,:,:]
 fig2 = Figure()
 ax = Axis(fig2[1,1], title="div")
 
 hm = contourf!(ax,grid.xf[1:grid.nx], grid.zc[1:grid.nz], semi.cache.div[1:grid.nx,1:grid.nz])
# hm = contourf!(ax,grid.xf[1:grid.nx], grid.zf[1:grid.nz], semi.cache.u[3,1:grid.nx,1:grid.nz])

 Colorbar(fig2[1,2], hm)
 display(fig2)