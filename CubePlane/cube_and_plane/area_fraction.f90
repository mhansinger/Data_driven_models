      program volume_plane_cube_intersection
      implicit none

! exercises the routines for calculating the 
! volume of the polyhedron resulting from the intersection of a plane and a cube

      character*40 :: string
      integer      :: i, j, k
      integer      :: npoints      ! number of points where the plane and cube intersect, a fascinating plot
      real*8       :: dx, volume, area_frac
      integer      :: nphi, ntheta, ndl
      real*8       :: phi_hi, phi_lo, phi_step, phi, &
                      theta_hi, theta_lo, theta_step, theta, &
                      dl_hi, dl_lo, dl_step, dl

! constants for converting between degrees and radians
      real*8, parameter :: pi     = 3.1415926535897932384d0, a2rad  = pi/180.0d0,  rad2a = 180.0d0/pi

! popular formats
10    format(1x,1p6e24.16)
11    format(1x,'area_',i2.2,'.dat')
12    format(1x,1f6.2)
13    format(1x,a,1pe12.4,a,1pe12.4,a,1pe12.4)


! length of a side of the unit cube
      dx = 1.0d0


! loop limits for dl, theta, and phi 
! dl runs from to to sqrt(dx^2 + dx^2 + dx^2) = sqrt(3), theta from 0 to 2*pi, phi from 0 to pi.
      dl_hi =  1.7d0
      dl_lo =  0.1d0
      ndl   =  17    ! 33  ! 161   ! number of dl, one movie frame per dl
      dl_step = 0.0d0
      if (ndl .ne. 1) dl_step = (dl_hi - dl_lo)/dfloat(ndl - 1)

      phi_hi   = 0.0d0   
      phi_lo   = 90.0d0  
      nphi     = 401     
      phi_step = 0.0d0
      if (nphi .ne. 1) phi_step = (phi_hi - phi_lo)/dfloat(nphi - 1)
       
      theta_hi   = 90.0d0   
      theta_lo   = 0.0d0    
      ntheta     = 401      
      theta_step = 0.0d0
      if (ntheta .ne. 1) theta_step = (theta_hi - theta_lo)/dfloat(ntheta - 1)


! loop over ray length dl - the radius vector
      do k = 1, ndl
       dl = dl_lo + dfloat(k-1)*dl_step

       write(6,*) 'working on dl ',dl

       write(string,11) k
       open(unit=2,file=string,status='unknown')
      ! write(2,12) dl

! loop over polar angle phi
      do i=1,nphi
       phi = phi_lo + dfloat(i-1)*phi_step
       phi = phi * a2rad

! loop over azimuthal angle theta
       do j=1,ntheta
        theta = theta_lo + dfloat(j-1)*theta_step
        theta = theta * a2rad

! ###################
! compute volume and report what we found

        call area_fraction(dl,theta,phi,volume,npoints,area_frac)
        volume = volume * dx**3
        !area_frac =  area_frac 


! TODO: area has to be filled!
! write what we found
        write(2,10) theta*rad2a, phi*rad2a, area_frac, volume, float(npoints)  ! Hansinger: replaced volume with area
! ###################

! end of theta loop
       enddo

! end of phi loop
! write a blank line for gnuplot
       write(2,*) ' '
      enddo

! end of dl loop
      close(unit=2)
      enddo

      end program volume_plane_cube_intersection





        subroutine volume_fraction(dl,theta,phi,volume,npoints)
        implicit none

! computes the volume of plane cube interface
!
! task 1 - find the plane-cube intersection points.
!          if this is all that was wanted, this routine would be a *lot* shorter.
!
! task 2 - form the list of face vertices.
!          this is the bulk of the routine. 
!
! task 3 - form the volume associated with each face.
!          relatively trivial once the face vertices are known
!
! it is useful to know the orderings in cube_path.pdf to understand this routine

! input:
! dl    = length of ray i.e., the radius from the origin
! theta = spherical coordinate angle theta of ray dl;  theta from 0 to 2*pi
! phi   = spherical coordinate angle phi of ray dl; phi from 0 to pi.

! output
! volume = volume of box cut by the intersection of the ray's normal plane and the cube
! npoints = number of intersection points; only values of 3, 4, 5, and 6 are possible.

! declare the pass
        integer  :: npoints
        real*8   :: dl, theta, phi, volume


! declare local variables
! how much to write
      integer, parameter :: iwrite = 0, idebug = 1


! ipath stores on which pathway an intersection was found
! ipath = 1 = light gray +x +z +y path
! ipath = 2 = gray +y +x +z path 
! ipath = 3 = black +z +y +x path
! ipath = 4 = dotted light gray, parallel to y-axis, path
! ipath = 5 = dotted gray, parallel to z-axis, path
! ipath = 6 = dotted black, parallel to x-axis, path

      integer :: ipath(12)


! for the plane-cube intersection points
! npoints is the number of intersections with the unit cube
! x_intersect, y_intersect, z_intersect store the coordinates of the intersection points 

      real*8  :: x_intersect(6), y_intersect(6), z_intersect(6)


! for the vertices
! nvert is the number of vertices, which includes both cube vertices (8) and intersection points (6 at maximum)
! x_vertex, y_vertex, z_vertex store the coordinates of the vertices

      integer :: nvert
      real*8  :: x_vertex(14), y_vertex(14), z_vertex(14)


! for the faces
! nfaces counts the number of faces with intersection, a maximum of 7 - six for the cube plus one for the plane
! face stores the face number
! nface_vertices stores how many vertices are with each face
! iverticies stores the vertices of eah face
! iface_mask is a convenient was of turning faces on or off

      integer :: nfaces, face(7), nface_vertices(7), ivertices(7,7), iface_mask(7)


! for the edges
! kedge stores the edge number

      integer :: iedge, jedge, kedge(6)


! others
      integer :: m, n, idum, jdum, kdum
      real*8  :: xn, yn, zn, mag, a, b, c, d, xx, yy, zz, &
                 lambda, xdum(8), ydum (8), zdum(8), area, dv, dl_max


! coordinates of the unit cube
      real*8, parameter  :: vx(8) = (/ 0.0d0, 1.0d0, 0.0d0, 0.0d0, 1.0d0, 1.0d0, 0.0d0, 1.0d0 /), &
                            vy(8) = (/ 0.0d0, 0.0d0, 1.0d0, 0.0d0, 0.0d0, 1.0d0, 1.0d0, 1.0d0 /), &
                            vz(8) = (/ 0.0d0, 0.0d0, 0.0d0, 1.0d0, 1.0d0, 0.0d0, 1.0d0, 1.0d0 /)


! for convenience only
      real*8, parameter :: pi     = 3.1415926535897932384d0, a2rad  = pi/180.0d0,  rad2a = 180.0d0/pi



! popular formats
10    format(1x,1p8e14.6)
11    format(1x,i4,1p3e10.2)
12    format(1x,2i4,1p3e10.2)
13    format(1x,a,i4,1p3e10.2)
14    format(1x,a,1pe12.4,a,1pe12.4,a,i3)


! initialize
      npoints = 0
      volume  = -1.0   

! check if there is an intersection at all
      dl_max  = (cos(theta) + sin(theta))*sin(phi) + cos(phi)
      if (dl .gt. dl_max) return


! coordinates of the normal vector of length dl
      xn =  dl * cos(theta) * sin(phi)
      yn =  dl * sin(theta) * sin(phi)
      zn =  dl * cos(phi)


! hessian normal form of the cutting plane, a*x + b*y + c*z = d
      mag = sqrt(xn*xn + yn*yn + zn*zn)
      a  = xn/mag ;  b = yn/mag ; c = zn/mag ; d = dl


! initialize all face and vertex data
      npoints = 0 ; nfaces = 0 ; face = 0 ; nface_vertices = 0 ; ivertices = 0
      x_vertex = 0.0d0 ; y_vertex = 0.0d0 ; z_vertex = 0.0d0
      ipath = 0 ; iface_mask = 0



! starting vertex v1 contributes to three faces - 1, 4, and 5
      nvert = 1
      x_vertex(nvert) = vx(1)  ; y_vertex(nvert) = vy(1)  ; z_vertex(nvert) = vz(1) 
      ipath(nvert) = 0 
      call add_vertex_to_face(1,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
      call add_vertex_to_face(4,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
      call add_vertex_to_face(5,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)


! face 7, the intersecting plane, is slightly special - don't add this vertex, but set it up
      nfaces = nfaces + 1 ; face(nfaces) = 7 ; iface_mask(7) = 1

      if (iwrite .eq. 1) write(6,13) 'adding vertex', nvert, x_vertex(nvert),y_vertex(nvert), z_vertex(nvert)


!-------------------------------------------------------------------------------------------------------------------     

! starting from vertex 1, first independent path is +x +z +y (light gray solid lines in cube_paths.pdf)
! an intersection can only occur once along this path, hence the triple if block

! check along +x 

      iedge = 1 ; jedge = 2
      call check_edge(iedge,jedge,vx,vy,vz,8,a,b,c,d,lambda,xx,yy,zz)

! there is a valid intersection 
      if (lambda .ge. 0.0d0 .and. lambda .le. 1.0d0) then
       if (iwrite .eq. 1) write(6,*) 'intersection along edges',iedge, jedge, lambda

! see if this vertex already exists
       call check_if_vertex_exists(x_intersect,y_intersect,z_intersect,npoints,xx,yy,zz,idum)

! if not, add this vertex to faces 1, 5, and 7
       if (idum .eq. 0) then
        npoints = npoints + 1       
        kedge(npoints) = 1
        x_intersect(npoints) = xx ; y_intersect(npoints) = yy ; z_intersect(npoints) = zz

        nvert = nvert + 1
        x_vertex(nvert) = x_intersect(npoints) ; y_vertex(nvert) = y_intersect(npoints) ; z_vertex(nvert) = z_intersect(npoints)
        ipath(nvert) = 1
        call add_vertex_to_face(1,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        call add_vertex_to_face(5,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        call add_vertex_to_face(7,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        if (iwrite .eq. 1) write(6,13) 'adding vertex', nvert, x_vertex(nvert),y_vertex(nvert), z_vertex(nvert)


! if needed, add end point vertices to faces 4 and 2
        if (lambda .eq. 0.0) then
         call add_vertex_to_face(4,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        else if (lambda .eq. 1.0) then
         call add_vertex_to_face(2,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        end if
       end if


! no intersection along +x, now check along +z
! vertex v2, contributes to faces 1, 2 and 5
! less commentary as the pattern emerges

      else

       nvert = nvert + 1
       x_vertex(nvert) = vx(jedge) ; y_vertex(nvert) = vy(jedge) ; z_vertex(nvert) = vz(jedge)
       ipath(nvert) = 1
       call add_vertex_to_face(1,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
       call add_vertex_to_face(2,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
       call add_vertex_to_face(5,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)

       if (iwrite .eq. 1) write(6,13) 'adding vertex', nvert, x_vertex(nvert), y_vertex(nvert), z_vertex(nvert)

       iedge = 2 ; jedge = 5
       call check_edge(iedge,jedge,vx,vy,vz,8,a,b,c,d,lambda,xx,yy,zz)

       if (lambda .ge. 0.0d0 .and. lambda .le. 1.0d0) then
        if (iwrite .eq. 1) write(6,*) 'intersection along edges',iedge, jedge, lambda

        call check_if_vertex_exists(x_intersect,y_intersect,z_intersect,npoints,xx,yy,zz,idum)

! add this vertex to faces 2, 5, and 7
        if (idum .eq. 0) then
         npoints = npoints + 1       
         kedge(npoints) = 2
         x_intersect(npoints) = xx ; y_intersect(npoints) = yy ; z_intersect(npoints) = zz

         nvert = nvert + 1
         x_vertex(nvert) = x_intersect(npoints) ; y_vertex(nvert) = y_intersect(npoints) ; z_vertex(nvert) = z_intersect(npoints)
         ipath(nvert) = 1
         call add_vertex_to_face(2,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
         call add_vertex_to_face(5,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
         call add_vertex_to_face(7,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
         if (iwrite .eq. 1) write(6,13) 'adding vertex', nvert, x_vertex(nvert),y_vertex(nvert), z_vertex(nvert)

! if needed, add end point vertices to faces 1 and 3
         if (lambda .eq. 0.0) then
          call add_vertex_to_face(1,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
         else if (lambda .eq. 1.0) then
          call add_vertex_to_face(3,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
         end if
        end if


! no intersection along +z either, now check +y 
! vertex v5, contributes to faces 2, 3 and 5

       else

        nvert = nvert + 1
        x_vertex(nvert) = vx(jedge) ; y_vertex(nvert) = vy(jedge) ; z_vertex(nvert) = vz(jedge)
        ipath(nvert) = 1
        call add_vertex_to_face(2,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        call add_vertex_to_face(3,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        call add_vertex_to_face(5,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        if (iwrite .eq. 1) write(6,13) 'adding vertex', nvert, x_vertex(nvert),y_vertex(nvert), z_vertex(nvert)

        iedge = 5 ; jedge = 8
        call check_edge(iedge,jedge,vx,vy,vz,8,a,b,c,d,lambda,xx,yy,zz)
         
        if (lambda .ge. 0.0d0 .and. lambda .le. 1.0d0) then
         if (iwrite .eq. 1) write(6,*) 'intersection along edges',iedge, jedge, lambda

          call check_if_vertex_exists(x_intersect,y_intersect,z_intersect,npoints,xx,yy,zz,idum)

! add this vertex to faces 2, 3, and 7
         if (idum .eq. 0) then
          npoints = npoints + 1       
          kedge(npoints) = 3
          x_intersect(npoints) = xx ; y_intersect(npoints) = yy ; z_intersect(npoints) = zz

          nvert = nvert + 1
          x_vertex(nvert) = x_intersect(npoints) ; y_vertex(nvert) = y_intersect(npoints) ; z_vertex(nvert) = z_intersect(npoints)
          ipath(nvert) = 1
          call add_vertex_to_face(2,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
          call add_vertex_to_face(3,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
          call add_vertex_to_face(7,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
          if (iwrite .eq. 1) write(6,13) 'adding vertex', nvert, x_vertex(nvert),y_vertex(nvert), z_vertex(nvert)

! if needed, add end point vertices to faces 5 and 6
          if (lambda .eq. 0.0) then
           call add_vertex_to_face(5,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
          else if (lambda .eq. 1.0) then
           call add_vertex_to_face(6,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
          end if
         end if

! end of +y lambda between 0 and 1 if block
        end if

! end of +z +y lambda between 0 and 1 if block
       end if

! end of +x lambda between 0 and 1 if block
      end if


! line intersection +x is parallel to y 
! light gray dashed line between v2 and v6, in this direction, in figure cube_paths.pdf

      iedge = 2 ; jedge = 6
      call check_edge(iedge,jedge,vx,vy,vz,8,a,b,c,d,lambda,xx,yy,zz)

      if (lambda .ge. 0.0d0 .and. lambda .le. 1.0d0) then
       if (iwrite .eq. 1) write(6,*) 'intersection along edges',iedge, jedge, lambda

       call check_if_vertex_exists(x_intersect,y_intersect,z_intersect,npoints,xx,yy,zz,idum)

! add this vertex to faces 1, 2, and 7 
       if (idum .eq. 0) then
        npoints = npoints + 1       
        kedge(npoints) = 4
        x_intersect(npoints) = xx ; y_intersect(npoints) = yy ; z_intersect(npoints) = zz

        nvert = nvert + 1
        x_vertex(nvert) = x_intersect(npoints) ; y_vertex(nvert) = y_intersect(npoints) ; z_vertex(nvert) = z_intersect(npoints)
        ipath(nvert) = 4
        call add_vertex_to_face(1,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        call add_vertex_to_face(2,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        call add_vertex_to_face(7,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        if (iwrite .eq. 1) write(6,13) 'adding vertex', nvert, x_vertex(nvert),y_vertex(nvert), z_vertex(nvert)


! if needed, add end point vertices to faces 5 and 6  
        if (lambda .eq. 0.0) then
         call add_vertex_to_face(5,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        else if (lambda .eq. 1.0) then
         call add_vertex_to_face(6,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        end if
       end if
      end if



!-------------------------------------------------------------------------------------------------------------------

! starting from vertex 1, second independent path is +y +x +z (gray solid lines in cube_paths.pdf)
! an intersection can only occur once along this path, hence the triple if block
! check +y

      iedge = 1 ; jedge = 3
      call check_edge(iedge,jedge,vx,vy,vz,8,a,b,c,d,lambda,xx,yy,zz)

      if (lambda .ge. 0.0d0 .and. lambda .le. 1.0d0) then
       if (iwrite .eq. 1) write(6,*) 'intersection along edges',iedge, jedge, lambda

       call check_if_vertex_exists(x_intersect,y_intersect,z_intersect,npoints,xx,yy,zz,idum)

! add this vertex to faces 1, 4, and 7 
       if (idum .eq. 0) then
        npoints = npoints + 1       
        kedge(npoints) = 5
        x_intersect(npoints) = xx ; y_intersect(npoints) = yy ; z_intersect(npoints) = zz

        nvert = nvert + 1
        x_vertex(nvert) = x_intersect(npoints) ; y_vertex(nvert) = y_intersect(npoints) ; z_vertex(nvert) = z_intersect(npoints)
        ipath(nvert) = 2
        call add_vertex_to_face(1,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        call add_vertex_to_face(4,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        call add_vertex_to_face(7,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        if (iwrite .eq. 1) write(6,13) 'adding vertex', nvert, x_vertex(nvert),y_vertex(nvert), z_vertex(nvert)

! if needed, add end point vertices to faces 5 and 6   
        if (lambda .eq. 0.0) then
         call add_vertex_to_face(5,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        else if (lambda .eq. 1.0) then
         call add_vertex_to_face(6,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        end if
       end if


! no intersection along +y, now check +x
! vertex v3 which contributes to faces 1, 4, and 6

      else

       nvert = nvert + 1 
       x_vertex(nvert) = vx(jedge) ; y_vertex(nvert) = vy(jedge) ; z_vertex(nvert) = vz(jedge)
       ipath(nvert) = 2
       call add_vertex_to_face(1,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
       call add_vertex_to_face(4,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
       call add_vertex_to_face(6,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
       if (iwrite .eq. 1) write(6,13) 'adding vertex', nvert, x_vertex(nvert),y_vertex(nvert), z_vertex(nvert)

       iedge = 3 ; jedge = 6
       call check_edge(iedge,jedge,vx,vy,vz,8,a,b,c,d,lambda,xx,yy,zz)
 
       if (lambda .ge. 0.0d0 .and. lambda .le. 1.0d0) then
        if (iwrite .eq. 1) write(6,*) 'intersection along edges',iedge, jedge, lambda

        call check_if_vertex_exists(x_intersect,y_intersect,z_intersect,npoints,xx,yy,zz,idum)

! add this vertex to faces 1, 6, and 7  
        if (idum .eq. 0) then
         npoints = npoints + 1       
         kedge(npoints) = 6
         x_intersect(npoints) = xx ; y_intersect(npoints) = yy ; z_intersect(npoints) = zz

         nvert = nvert + 1
         x_vertex(nvert) = x_intersect(npoints) ; y_vertex(nvert) = y_intersect(npoints) ; z_vertex(nvert) = z_intersect(npoints)
         ipath(nvert) = 2
         call add_vertex_to_face(1,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
         call add_vertex_to_face(6,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
         call add_vertex_to_face(7,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
         if (iwrite .eq. 1) write(6,13) 'adding vertex', nvert, x_vertex(nvert),y_vertex(nvert), z_vertex(nvert)

! if needed, add end point vertices to faces 4 and 2  
         if (lambda .eq. 0.0) then
          call add_vertex_to_face(4,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
         else if (lambda .eq. 1.0) then
          call add_vertex_to_face(2,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
         end if
        end if


! no intersection along +x, now check +z
! meaning we pick up vertex v6 which contributes to faces 1, 2, and  6

       else

        nvert = nvert + 1
        x_vertex(nvert) = vx(jedge) ; y_vertex(nvert) = vy(jedge) ; z_vertex(nvert) = vz(jedge)
        ipath(nvert) = 2
        call add_vertex_to_face(1,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        call add_vertex_to_face(2,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        call add_vertex_to_face(6,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        if (iwrite .eq. 1) write(6,13) 'adding vertex', nvert, x_vertex(nvert),y_vertex(nvert), z_vertex(nvert)

        iedge = 6 ; jedge = 8
        call check_edge(iedge,jedge,vx,vy,vz,8,a,b,c,d,lambda,xx,yy,zz)

        if (lambda .ge. 0.0d0 .and. lambda .le. 1.0d0) then
         if (iwrite .eq. 1) write(6,*) 'intersection along edges',iedge, jedge, lambda

         call check_if_vertex_exists(x_intersect,y_intersect,z_intersect,npoints,xx,yy,zz,idum)

! add this vertex to faces 2, 6, and 7 
         if (idum .eq. 0) then
          npoints = npoints + 1       
          kedge(npoints) = 7
          x_intersect(npoints) = xx ; y_intersect(npoints) = yy ; z_intersect(npoints) = zz
          nvert = nvert + 1
          x_vertex(nvert) = x_intersect(npoints) ; y_vertex(nvert) = y_intersect(npoints) ; z_vertex(nvert) = z_intersect(npoints)
          ipath(nvert) = 2
          call add_vertex_to_face(2,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
          call add_vertex_to_face(6,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
          call add_vertex_to_face(7,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
          if (iwrite .eq. 1) write(6,13) 'adding vertex', nvert, x_vertex(nvert),y_vertex(nvert), z_vertex(nvert)

! if needed, add end point vertices to faces 1 and 3 
          if (lambda .eq. 0.0) then
           call add_vertex_to_face(1,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
          else if (lambda .eq. 1.0) then
           call add_vertex_to_face(3,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
          end if
         end if

! end of +z lambda between 0 and 1 if block
        end if

! end of +x +z lambda between 0 and 1 if block
       end if

! end of +y +x +z lambda between 0 and 1 if block
      end if


! line intersections +y parallel to +z
! gray dashed line between v3 and v7 in figure cube_paths.pdf
      iedge = 3 ; jedge = 7
      call check_edge(iedge,jedge,vx,vy,vz,8,a,b,c,d,lambda,xx,yy,zz)
      if (lambda .ge. 0.0d0 .and. lambda .le. 1.0d0) then
       if (iwrite .eq. 1) write(6,*) 'intersection along edges',iedge, jedge, lambda

       call check_if_vertex_exists(x_intersect,y_intersect,z_intersect,npoints,xx,yy,zz,idum)

! add this vertex to faces 2, 6, and 7 
       if (idum .eq. 0) then
        npoints = npoints + 1       
        kedge(npoints) = 8
        x_intersect(npoints) = xx ; y_intersect(npoints) = yy ; z_intersect(npoints) = zz

        nvert = nvert + 1
        x_vertex(nvert) = x_intersect(npoints) ; y_vertex(nvert) = y_intersect(npoints) ; z_vertex(nvert) = z_intersect(npoints)
        ipath(nvert) = 5
        call add_vertex_to_face(4,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        call add_vertex_to_face(6,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        call add_vertex_to_face(7,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        if (iwrite .eq. 1) write(6,13) 'adding vertex', nvert, x_vertex(nvert),y_vertex(nvert), z_vertex(nvert)

! if needed, add end point vertices to faces 1 and 3   
        if (lambda .eq. 0.0) then
         call add_vertex_to_face(1,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        else if (lambda .eq. 1.0) then
         call add_vertex_to_face(3,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        end if
       end if
      end if



!-------------------------------------------------------------------------------------------------------------------     

! starting from vertex 1, third independent path is +z +y +x (black solid lines in cube_paths.pdf)
! an intersection can only occur once along this path, hence the triple if block
! check +z
      iedge = 1 ; jedge = 4
      call check_edge(iedge,jedge,vx,vy,vz,8,a,b,c,d,lambda,xx,yy,zz)

      if (lambda .ge. 0.0d0 .and. lambda .le. 1.0d0) then
       if (iwrite .eq. 1) write(6,*) 'intersection along edges',iedge, jedge, lambda

       call check_if_vertex_exists(x_intersect,y_intersect,z_intersect,npoints,xx,yy,zz,idum)

! add this vertex to faces 4, 5, and 7 
       if (idum .eq. 0) then
        npoints = npoints + 1       
        kedge(npoints) = 9
        x_intersect(npoints) = xx ; y_intersect(npoints) = yy ; z_intersect(npoints) = zz

        nvert = nvert + 1
        x_vertex(nvert) = x_intersect(npoints) ; y_vertex(nvert) = y_intersect(npoints) ; z_vertex(nvert) = z_intersect(npoints)
        ipath(nvert) = 3
        call add_vertex_to_face(4,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        call add_vertex_to_face(5,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        call add_vertex_to_face(7,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        if (iwrite .eq. 1) write(6,13) 'adding vertex', nvert, x_vertex(nvert),y_vertex(nvert), z_vertex(nvert)

! if needed, add end point vertices to faces 1 and 3
        if (lambda .eq. 0.0) then
         call add_vertex_to_face(1,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        else if (lambda .eq. 1.0) then
         call add_vertex_to_face(3,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        end if
       end if


! no intersection along +z, now check +y
! meaning we pick up vertex v4 which contributes to faces 3, 4, and 5

      else

       nvert = nvert + 1
       x_vertex(nvert) = vx(jedge) ; y_vertex(nvert) = vy(jedge) ; z_vertex(nvert) = vz(jedge)
       ipath(nvert) = 3
       call add_vertex_to_face(3,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
       call add_vertex_to_face(4,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
       call add_vertex_to_face(5,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
       if (iwrite .eq. 1) write(6,13) 'adding vertex', nvert, x_vertex(nvert),y_vertex(nvert), z_vertex(nvert)

       iedge = 4 ; jedge = 7
       call check_edge(iedge,jedge,vx,vy,vz,8,a,b,c,d,lambda,xx,yy,zz)

       if (lambda .ge. 0.0d0 .and. lambda .le. 1.0d0) then
        if (iwrite .eq. 1) write(6,*) 'intersection along edges',iedge, jedge, lambda

        call check_if_vertex_exists(x_intersect,y_intersect,z_intersect,npoints,xx,yy,zz,idum)

! add this vertex to faces 3, 4, and 7  
        if (idum .eq. 0) then
         npoints = npoints + 1       
         kedge(npoints) = 10
         x_intersect(npoints) = xx ; y_intersect(npoints) = yy ; z_intersect(npoints) = zz

         nvert = nvert + 1
         x_vertex(nvert) = x_intersect(npoints) ; y_vertex(nvert) = y_intersect(npoints) ; z_vertex(nvert) = z_intersect(npoints)
         ipath(nvert) = 3
         call add_vertex_to_face(3,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
         call add_vertex_to_face(4,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
         call add_vertex_to_face(7,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
         if (iwrite .eq. 1) write(6,13) 'adding vertex', nvert, x_vertex(nvert),y_vertex(nvert), z_vertex(nvert)

! if needed, add end point vertices to faces 5 and 6 
         if (lambda .eq. 0.0) then
          call add_vertex_to_face(5,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
         else if (lambda .eq. 1.0) then
          call add_vertex_to_face(6,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
         end if
        end if


! no intersection along +y, now check +x
! we pick up vertex v7 which contributes to faces 3, 4, and 6

       else

        nvert = nvert + 1
        x_vertex(nvert) = vx(jedge) ; y_vertex(nvert) = vy(jedge) ; z_vertex(nvert) = vz(jedge)
        ipath(nvert) = 3
        call add_vertex_to_face(3,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        call add_vertex_to_face(4,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        call add_vertex_to_face(6,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        if (iwrite .eq. 1) write(6,13) 'adding vertex', nvert, x_vertex(nvert),y_vertex(nvert), z_vertex(nvert)

        iedge = 7 ; jedge = 8
        call check_edge(iedge,jedge,vx,vy,vz,8,a,b,c,d,lambda,xx,yy,zz)

        if (lambda .ge. 0.0d0 .and. lambda .le. 1.0d0) then
         if (iwrite .eq. 1) write(6,*) 'intersection along edges',iedge, jedge, lambda

         call check_if_vertex_exists(x_intersect,y_intersect,z_intersect,npoints,xx,yy,zz,idum)

! add this vertex to faces 3, 6, and 7 
         if (idum .eq. 0) then
          npoints = npoints + 1       
          kedge(npoints) = 11
          x_intersect(npoints) = xx ; y_intersect(npoints) = yy ; z_intersect(npoints) = zz

          nvert = nvert + 1
          x_vertex(nvert) = x_intersect(npoints) ; y_vertex(nvert) = y_intersect(npoints) ; z_vertex(nvert) = z_intersect(npoints)
          ipath(nvert) = 3
          call add_vertex_to_face(3,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
          call add_vertex_to_face(6,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
          call add_vertex_to_face(7,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
          if (iwrite .eq. 1) write(6,13) 'adding vertex', nvert, x_vertex(nvert),y_vertex(nvert), z_vertex(nvert)

! if needed, add end point vertices to faces 4 and 2    
          if (lambda .eq. 0.0) then
           call add_vertex_to_face(4,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
          else if (lambda .eq. 1.0) then
           call add_vertex_to_face(2,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
          end if
         end if

! end of +x lambda between 0 and 1 if block
        end if

! end of +y +x lambda between 0 and 1 if block
       end if

! end of +z +y +x lambda between 0 and 1 if block
      end if


! line intersection +z parallel to x
! black dashed line between v4 and v5 in figure cube_paths.pdf
      iedge = 4 ; jedge = 5
      call check_edge(iedge,jedge,vx,vy,vz,8,a,b,c,d,lambda,xx,yy,zz)

      if (lambda .ge. 0.0d0 .and. lambda .le. 1.0d0) then
       if (iwrite .eq. 1) write(6,*) 'intersection along edges',iedge, jedge, lambda

       call check_if_vertex_exists(x_intersect,y_intersect,z_intersect,npoints,xx,yy,zz,idum)

! add this vertex to faces 3, 5, and 7
       if (idum .eq. 0) then
        npoints = npoints + 1       
        kedge(npoints) = 12
        x_intersect(npoints) = xx ; y_intersect(npoints) = yy ; z_intersect(npoints) = zz

        nvert = nvert + 1
        x_vertex(nvert) = x_intersect(npoints) ; y_vertex(nvert) = y_intersect(npoints) ; z_vertex(nvert) = z_intersect(npoints)
        ipath(nvert) = 6
        call add_vertex_to_face(3,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        call add_vertex_to_face(5,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        call add_vertex_to_face(7,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        if (iwrite .eq. 1) write(6,13) 'adding vertex', nvert, x_vertex(nvert),y_vertex(nvert), z_vertex(nvert)

! if needed, add end point vertices to faces 4 and 2    
        if (lambda .eq. 0.0) then
         call add_vertex_to_face(4,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        else if (lambda .eq. 1.0) then
         call add_vertex_to_face(2,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        end if
       end if
      end if



! whew. now we have a list of potential faces and vertices
!-------------------------------------------------------------------------------------------------------------------     

      if (iwrite .eq. 2) then
       write(6,*)
       write(6,14) 'phi=',phi*rad2a, ' theta=',theta*rad2a, ' npoints =',npoints
       do n=1,npoints
        write(6,10) x_intersect(n),y_intersect(n),z_intersect(n)
       enddo
      end if
      if (iwrite .eq. 1) then
       write(6,*)
       write(6,*) 'vertices'
       do m=1,nvert
        write(6,101) m, ipath(m), x_vertex(m), y_vertex(m), z_vertex(m)
101     format(1x,2i4,1p3e12.4)
       enddo
       write(6,*)
       write(6,*) 'nfaces =',nfaces
       write(6,102) 'iface_mask =',iface_mask
102    format(1x,a,7i2)
       do m=1,nfaces
        write(6,*) face(m),nface_vertices(face(m))
        write(6,*) (ivertices(n,face(m)), n=1,nface_vertices(face(m)))
       enddo
       write(6,*)
      end if


!-------------------------------------------------------------------------------------------------------------------     

! now address various pathologies
! the utility of face_mask is thus revealed.
!
! in the degenerate case, the plane intersects one of the faces.
! remove the double counting here.
! construct a unique integer from the vertex numbers of the intersecting plane
     idum = 0
     do n=1,nface_vertices(7)
      idum = idum + 10**(n-1)*ivertices(n,7)
     end do

! construct a unique integer from the vertex numbers of the other faces
     do m= 1, nfaces
      jdum = 0
      do n=1,nface_vertices(face(m))
       jdum = jdum + 10**(n-1)*ivertices(n,face(m))
      enddo

! zero the face_mask if the two integers match
      if (jdum .eq. idum .and. face(m) .ne. 7) then
       iface_mask(face(m)) = 0
       if (iwrite .eq. 1) then
        write(6,*) 'degenerate face'
        write(6,*) 'removing face ',m,face(m)
        write(6,*) (ivertices(n,7), n=1,nface_vertices(7))
        write(6,*) (ivertices(n,face(m)), n=1,nface_vertices(face(m)))
       end if
      end if

! another case to address is no intersection at all and only cube vertices.
! remove the faces where the number of vertices is less than 3
      if (iface_mask(face(m)) .eq. 1 .and. nface_vertices(face(m)) .lt. 3) then
       iface_mask(face(m)) = 0
       if (iwrite .eq. 1) then
        write(6,*) 'face with less than 3 vertices'
        write(6,*) 'removing face ',m,face(m),nface_vertices(face(m))
       end if
      end if
     end do


!-------------------------------------------------------------------------------------------------------------------     


! compute the volume 
! if we got any intersection

     if (npoints .gt. 0) then
      volume = 0.0d0

! loop over valid faces
      do m = 4,nfaces
       if (iface_mask(face(m)) .eq. 1) then


! the vertices of the intersecting plane, by construction, are in clockwise order.
! the cube face intersections do not have a consistent winding; 
! we put them in clockwise order by reversing paths 

! face 1
! reverse order of vertexes along path 2, dark gray in cube_paths.pdf

       if (face(m) .eq. 1) then
        idum = 0
        do n = 1, nface_vertices(face(m))
         kdum = ivertices(n,face(m))   
         if (ipath(kdum) .eq. 0 .or. ipath(kdum) .eq. 1 .or. ipath(kdum) .eq. 4) then
          idum = idum + 1
          xdum(idum) = x_vertex(kdum) ; ydum(idum) = y_vertex(kdum) ; zdum(idum) = z_vertex(kdum)
         end if
        end do
        do n = nface_vertices(face(m)), 1, -1
         kdum = ivertices(n,face(m)) 
         if (ipath(kdum) .eq. 2) then
          idum = idum + 1
          xdum(idum) = x_vertex(kdum) ; ydum(idum) = y_vertex(kdum) ; zdum(idum) = z_vertex(kdum)
         end if
        end do

        if (idebug .eq. 1) then
         call irregular_polygon_area(xdum,ydum,zdum,nface_vertices(face(m)),area)
         if (area .eq. 0.0) then
          write(6,14) 'dl =',dl, ' phi  =',phi*rad2a, ' theta=',theta*rad2a
          write(6,11) (n,xdum(n),ydum(n),zdum(n), n=1,nface_vertices(face(m)))
          write(6,*) 'zero area face 1 - bad ordering'
         end if
        end if


! face 2
! reverse order of vertexes along paths 2 and 4, dark gray and dotted light gray in cube_paths.pdf

       else if (face(m) .eq. 2) then
        idum = 0
        do n = 1, nface_vertices(face(m))
         kdum = ivertices(n,face(m))
         if (ipath(kdum) .eq. 1 ) then
          idum = idum + 1
          xdum(idum) = x_vertex(kdum) ; ydum(idum) = y_vertex(kdum) ; zdum(idum) = z_vertex(kdum)
         end if
        end do
        do n = nface_vertices(face(m)), 1, -1
         kdum = ivertices(n,face(m))
         if (ipath(kdum) .eq. 2 .or. ipath(kdum) .eq. 4 ) then
          idum = idum + 1
          xdum(idum) = x_vertex(kdum) ; ydum(idum) = y_vertex(kdum) ; zdum(idum) = z_vertex(kdum)
         end if
        end do
        if (idebug .eq. 1) then
         call irregular_polygon_area(xdum,ydum,zdum,nface_vertices(face(m)),area)
         if (area .eq. 0.0) then
          write(6,14) 'dl =',dl, ' phi  =',phi*rad2a, ' theta=',theta*rad2a
          write(6,11) (n,xdum(n),ydum(n),zdum(n), n=1,nface_vertices(face(m)))
          write(6,*) 'zero area face 2 - bad ordering'
         end if
        end if


! face 3
! reverse order of vertexes along path 3 , black in cube_paths.pdf 

       else if (face(m) .eq. 3) then
        idum = 0
        do n = 1, nface_vertices(face(m))
         kdum = ivertices(n,face(m))
         if (ipath(kdum) .eq. 1 .or. ipath(kdum) .eq. 6 ) then
          idum = idum + 1
          xdum(idum) = x_vertex(kdum) ; ydum(idum) = y_vertex(kdum) ; zdum(idum) = z_vertex(kdum)
         end if
        end do
        do n = nface_vertices(face(m)), 1, -1
         kdum = ivertices(n,face(m))
         if (ipath(kdum) .eq. 3 ) then
          idum = idum + 1
          xdum(idum) = x_vertex(kdum) ; ydum(idum) = y_vertex(kdum) ; zdum(idum) = z_vertex(kdum)
         end if
        end do
        if (idebug .eq. 1) then
         call irregular_polygon_area(xdum,ydum,zdum,nface_vertices(face(m)),area)
         if (area .eq. 0.0) then
          write(6,14) 'dl =',dl, ' phi  =',phi*rad2a, ' theta=',theta*rad2a
          write(6,11) (n,xdum(n),ydum(n),zdum(n), n=1,nface_vertices(face(m)))
          write(6,*) 'zero area face 3'
         end if
        end if


! face 4
! reverse order of vertexes along path 3 , black in cube_paths.pdf 

       else if (face(m) .eq. 4) then
        idum = 0
        do n = 1, nface_vertices(face(m))
         kdum = ivertices(n,face(m))
         if (ipath(kdum) .eq. 0 .or. ipath(kdum) .eq. 2 .or. ipath(kdum) .eq. 5 ) then
          idum = idum + 1
          xdum(idum) = x_vertex(kdum) ; ydum(idum) = y_vertex(kdum) ; zdum(idum) = z_vertex(kdum)
         end if
        end do
        do n = nface_vertices(face(m)), 1, -1
         kdum = ivertices(n,face(m))
         if (ipath(kdum) .eq. 3 ) then
          idum = idum + 1
          xdum(idum) = x_vertex(kdum) ; ydum(idum) = y_vertex(kdum) ; zdum(idum) = z_vertex(kdum)
         end if
        end do
        if (idebug .eq. 1) then
         call irregular_polygon_area(xdum,ydum,zdum,nface_vertices(face(m)),area)
         if (area .eq. 0.0) then
          write(6,14) 'dl =',dl, ' phi  =',phi*rad2a, ' theta=',theta*rad2a
          write(6,11) (n,xdum(n),ydum(n),zdum(n), n=1,nface_vertices(face(m)))
          write(6,*) 'zero area face 4'
         end if
        end if


! face 5
! reverse order of vertexes along paths 3 and 6, black and dark dashed in cube_paths.pdf 

       else if (face(m) .eq. 5) then
        idum = 0
        do n = 1, nface_vertices(face(m))
         kdum = ivertices(n,face(m))
         if (ipath(kdum) .eq. 0 .or. ipath(kdum) .eq. 1 ) then
          idum = idum + 1
          xdum(idum) = x_vertex(kdum) ; ydum(idum) = y_vertex(kdum) ; zdum(idum) = z_vertex(kdum)
         end if
        end do
        do n = nface_vertices(face(m)), 1, -1
         kdum = ivertices(n,face(m))
         if (ipath(kdum) .eq. 3 .or. ipath(kdum) .eq. 6 ) then
          idum = idum + 1
          xdum(idum) = x_vertex(kdum) ; ydum(idum) = y_vertex(kdum) ; zdum(idum) = z_vertex(kdum)
         end if
        end do
        if (idebug .eq. 1) then
         call irregular_polygon_area(xdum,ydum,zdum,nface_vertices(face(m)),area)
         if (area .eq. 0.0) then
          write(6,14) 'dl =',dl, ' phi  =',phi*rad2a, ' theta=',theta*rad2a
          write(6,11) (n,xdum(n),ydum(n),zdum(n), n=1,nface_vertices(face(m)))
          write(6,*) 'zero area face 5'
         end if
        end if


! face 6
! reverse order of vertexes along paths 3 and 5, black and gray in cube_paths.pdf 

       else if (face(m) .eq. 6) then
        idum = 0
        do n = 1, nface_vertices(face(m))
         kdum = ivertices(n,face(m))
         if (ipath(kdum) .eq. 2 ) then
          idum = idum + 1
          xdum(idum) = x_vertex(kdum) ; ydum(idum) = y_vertex(kdum) ; zdum(idum) = z_vertex(kdum)
         end if
        end do
        do n = nface_vertices(face(m)), 1, -1
         kdum = ivertices(n,face(m))
         if (ipath(kdum) .eq. 3 .or. ipath(kdum) .eq. 5 ) then
          idum = idum + 1
          xdum(idum) = x_vertex(kdum) ; ydum(idum) = y_vertex(kdum) ; zdum(idum) = z_vertex(kdum)
         end if
        end do
        if (idebug .eq. 1) then
         call irregular_polygon_area(xdum,ydum,zdum,nface_vertices(face(m)),area)
         if (area .eq. 0.0) then
          write(6,14) 'dl =',dl, ' phi  =',phi*rad2a, ' theta=',theta*rad2a
          write(6,11) (n,xdum(n),ydum(n),zdum(n), n=1,nface_vertices(face(m)))
          write(6,*) 'zero area face 6'
         end if
        end if


! face 7 is the intersecting plane and the final face
! they are in clockwise order by construction

       else if (face(m) .eq. 7) then
        do n = 1, nface_vertices(face(m))
         kdum = ivertices(n,face(m))
         xdum(n) = x_vertex(kdum) ; ydum(n) = y_vertex(kdum) ; zdum(n) = z_vertex(kdum)
        end do
       end if


! volume of this polyhedra
! choose vertex 1 as the reference height of the pyramids.
! faces 1, 4, and 5 then contribute zero volume because their heights are zero.
! only faces 2, 3, 6 and 7 contribute.

        call pyramid_volume(xdum,ydum,zdum,nface_vertices(face(m)),x_vertex(1),y_vertex(1),z_vertex(1),dv)

        volume = volume + dv

        if (iwrite .eq. 1) then
         write(6,12) m,face(m),volume, dv
        else if (iwrite .eq. 2) then
         write(6,*) ' face ', face(m),'  nvertices', nface_vertices(face(m)) 
         write(6,'(1pe12.4,1pe12.4,1pe12.4)') (xdum(n),ydum(n),zdum(n), n = 1, nface_vertices(face(m)) )
         write(6,'(a,1pe12.4)') 'dv ',dv
        end if


! end of loop over valid faces
       end if
      enddo


! check the volume is between 0.0 and 1.0
      if (iwrite .eq. 1) write(6,*) 'volume =',volume

      if (volume .le. 0.0d0) then
       write(6,*)
       write(6,*) 'volume is negative',volume
       write(6,10) theta*rad2a, phi*rad2a, dl, volume
       write(6,*) npoints
       write(6,12) (n, kedge(n), x_intersect(n), y_intersect(n), z_intersect(n), n=1,npoints)
!       stop 'bad volume'
      end if

      if (volume .gt. 1.0d0) then
       write(6,*)
       write(6,*) 'volume of unit cube > 1 ',volume
       write(6,10) theta*rad2a, phi*rad2a, dl, volume
       write(6,*) npoints
       write(6,12) (n, kedge(n), x_intersect(n), y_intersect(n), z_intersect(n), n=1,npoints)
!       stop 'bad volume'
      end if

! end of npoints if block on the volume calculation
      end if

      return
      end subroutine volume_fraction




!#############################
! added Hansinger

        subroutine area_fraction(dl,theta,phi,volume,npoints,area)
        implicit none

! computes the volume of plane cube interface
!
! task 1 - find the plane-cube intersection points.
!          if this is all that was wanted, this routine would be a *lot* shorter.
!
! task 2 - form the list of face vertices.
!          this is the bulk of the routine. 
!
! task 3 - form the volume associated with each face.
!          relatively trivial once the face vertices are known
!
! it is useful to know the orderings in cube_path.pdf to understand this routine

! input:
! dl    = length of ray i.e., the radius from the origin
! theta = spherical coordinate angle theta of ray dl;  theta from 0 to 2*pi
! phi   = spherical coordinate angle phi of ray dl; phi from 0 to pi.

! output
! area = intersection area relative to side of the cube
! volume = volume of box cut by the intersection of the ray's normal plane and the cube
! npoints = number of intersection points; only values of 3, 4, 5, and 6 are possible.

! declare the pass
        integer  :: npoints
        real*8   :: dl, theta, phi, volume, area_frac


! declare local variables
! how much to write
      integer, parameter :: iwrite = 0, idebug = 1


! ipath stores on which pathway an intersection was found
! ipath = 1 = light gray +x +z +y path
! ipath = 2 = gray +y +x +z path 
! ipath = 3 = black +z +y +x path
! ipath = 4 = dotted light gray, parallel to y-axis, path
! ipath = 5 = dotted gray, parallel to z-axis, path
! ipath = 6 = dotted black, parallel to x-axis, path

      integer :: ipath(12)


! for the plane-cube intersection points
! npoints is the number of intersections with the unit cube
! x_intersect, y_intersect, z_intersect store the coordinates of the intersection points 

      real*8  :: x_intersect(6), y_intersect(6), z_intersect(6)


! for the vertices
! nvert is the number of vertices, which includes both cube vertices (8) and intersection points (6 at maximum)
! x_vertex, y_vertex, z_vertex store the coordinates of the vertices

      integer :: nvert
      real*8  :: x_vertex(14), y_vertex(14), z_vertex(14)


! for the faces
! nfaces counts the number of faces with intersection, a maximum of 7 - six for the cube plus one for the plane
! face stores the face number
! nface_vertices stores how many vertices are with each face
! iverticies stores the vertices of eah face
! iface_mask is a convenient was of turning faces on or off

      integer :: nfaces, face(7), nface_vertices(7), ivertices(7,7), iface_mask(7)


! for the edges
! kedge stores the edge number

      integer :: iedge, jedge, kedge(6)


! others
      integer :: m, n, idum, jdum, kdum
      real*8  :: xn, yn, zn, mag, a, b, c, d, xx, yy, zz, &
                 lambda, xdum(8), ydum (8), zdum(8), area, dv, dl_max


! coordinates of the unit cube
      real*8, parameter  :: vx(8) = (/ 0.0d0, 1.0d0, 0.0d0, 0.0d0, 1.0d0, 1.0d0, 0.0d0, 1.0d0 /), &
                            vy(8) = (/ 0.0d0, 0.0d0, 1.0d0, 0.0d0, 0.0d0, 1.0d0, 1.0d0, 1.0d0 /), &
                            vz(8) = (/ 0.0d0, 0.0d0, 0.0d0, 1.0d0, 1.0d0, 0.0d0, 1.0d0, 1.0d0 /)


! for convenience only
      real*8, parameter :: pi     = 3.1415926535897932384d0, a2rad  = pi/180.0d0,  rad2a = 180.0d0/pi



! popular formats
10    format(1x,1p8e14.6)
11    format(1x,i4,1p3e10.2)
12    format(1x,2i4,1p3e10.2)
13    format(1x,a,i4,1p3e10.2)
14    format(1x,a,1pe12.4,a,1pe12.4,a,i3)


! initialize
      npoints = 0
      volume  = -1.0 
      area_frac = -1.0 ! added Hansinger

! check if there is an intersection at all
      dl_max  = (cos(theta) + sin(theta))*sin(phi) + cos(phi)
      if (dl .gt. dl_max) return


! coordinates of the normal vector of length dl
      xn =  dl * cos(theta) * sin(phi)
      yn =  dl * sin(theta) * sin(phi)
      zn =  dl * cos(phi)


! hessian normal form of the cutting plane, a*x + b*y + c*z = d
      mag = sqrt(xn*xn + yn*yn + zn*zn)
      a  = xn/mag ;  b = yn/mag ; c = zn/mag ; d = dl


! initialize all face and vertex data
      npoints = 0 ; nfaces = 0 ; face = 0 ; nface_vertices = 0 ; ivertices = 0
      x_vertex = 0.0d0 ; y_vertex = 0.0d0 ; z_vertex = 0.0d0
      ipath = 0 ; iface_mask = 0



! starting vertex v1 contributes to three faces - 1, 4, and 5
      nvert = 1
      x_vertex(nvert) = vx(1)  ; y_vertex(nvert) = vy(1)  ; z_vertex(nvert) = vz(1) 
      ipath(nvert) = 0 
      call add_vertex_to_face(1,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
      call add_vertex_to_face(4,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
      call add_vertex_to_face(5,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)


! face 7, the intersecting plane, is slightly special - don't add this vertex, but set it up
      nfaces = nfaces + 1 ; face(nfaces) = 7 ; iface_mask(7) = 1

      if (iwrite .eq. 1) write(6,13) 'adding vertex', nvert, x_vertex(nvert),y_vertex(nvert), z_vertex(nvert)


!-------------------------------------------------------------------------------------------------------------------     

! starting from vertex 1, first independent path is +x +z +y (light gray solid lines in cube_paths.pdf)
! an intersection can only occur once along this path, hence the triple if block

! check along +x 

      iedge = 1 ; jedge = 2
      call check_edge(iedge,jedge,vx,vy,vz,8,a,b,c,d,lambda,xx,yy,zz)

! there is a valid intersection 
      if (lambda .ge. 0.0d0 .and. lambda .le. 1.0d0) then
       if (iwrite .eq. 1) write(6,*) 'intersection along edges',iedge, jedge, lambda

! see if this vertex already exists
       call check_if_vertex_exists(x_intersect,y_intersect,z_intersect,npoints,xx,yy,zz,idum)

! if not, add this vertex to faces 1, 5, and 7
       if (idum .eq. 0) then
        npoints = npoints + 1       
        kedge(npoints) = 1
        x_intersect(npoints) = xx ; y_intersect(npoints) = yy ; z_intersect(npoints) = zz

        nvert = nvert + 1
        x_vertex(nvert) = x_intersect(npoints) ; y_vertex(nvert) = y_intersect(npoints) ; z_vertex(nvert) = z_intersect(npoints)
        ipath(nvert) = 1
        call add_vertex_to_face(1,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        call add_vertex_to_face(5,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        call add_vertex_to_face(7,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        if (iwrite .eq. 1) write(6,13) 'adding vertex', nvert, x_vertex(nvert),y_vertex(nvert), z_vertex(nvert)


! if needed, add end point vertices to faces 4 and 2
        if (lambda .eq. 0.0) then
         call add_vertex_to_face(4,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        else if (lambda .eq. 1.0) then
         call add_vertex_to_face(2,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        end if
       end if


! no intersection along +x, now check along +z
! vertex v2, contributes to faces 1, 2 and 5
! less commentary as the pattern emerges

      else

       nvert = nvert + 1
       x_vertex(nvert) = vx(jedge) ; y_vertex(nvert) = vy(jedge) ; z_vertex(nvert) = vz(jedge)
       ipath(nvert) = 1
       call add_vertex_to_face(1,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
       call add_vertex_to_face(2,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
       call add_vertex_to_face(5,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)

       if (iwrite .eq. 1) write(6,13) 'adding vertex', nvert, x_vertex(nvert), y_vertex(nvert), z_vertex(nvert)

       iedge = 2 ; jedge = 5
       call check_edge(iedge,jedge,vx,vy,vz,8,a,b,c,d,lambda,xx,yy,zz)

       if (lambda .ge. 0.0d0 .and. lambda .le. 1.0d0) then
        if (iwrite .eq. 1) write(6,*) 'intersection along edges',iedge, jedge, lambda

        call check_if_vertex_exists(x_intersect,y_intersect,z_intersect,npoints,xx,yy,zz,idum)

! add this vertex to faces 2, 5, and 7
        if (idum .eq. 0) then
         npoints = npoints + 1       
         kedge(npoints) = 2
         x_intersect(npoints) = xx ; y_intersect(npoints) = yy ; z_intersect(npoints) = zz

         nvert = nvert + 1
         x_vertex(nvert) = x_intersect(npoints) ; y_vertex(nvert) = y_intersect(npoints) ; z_vertex(nvert) = z_intersect(npoints)
         ipath(nvert) = 1
         call add_vertex_to_face(2,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
         call add_vertex_to_face(5,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
         call add_vertex_to_face(7,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
         if (iwrite .eq. 1) write(6,13) 'adding vertex', nvert, x_vertex(nvert),y_vertex(nvert), z_vertex(nvert)

! if needed, add end point vertices to faces 1 and 3
         if (lambda .eq. 0.0) then
          call add_vertex_to_face(1,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
         else if (lambda .eq. 1.0) then
          call add_vertex_to_face(3,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
         end if
        end if


! no intersection along +z either, now check +y 
! vertex v5, contributes to faces 2, 3 and 5

       else

        nvert = nvert + 1
        x_vertex(nvert) = vx(jedge) ; y_vertex(nvert) = vy(jedge) ; z_vertex(nvert) = vz(jedge)
        ipath(nvert) = 1
        call add_vertex_to_face(2,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        call add_vertex_to_face(3,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        call add_vertex_to_face(5,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        if (iwrite .eq. 1) write(6,13) 'adding vertex', nvert, x_vertex(nvert),y_vertex(nvert), z_vertex(nvert)

        iedge = 5 ; jedge = 8
        call check_edge(iedge,jedge,vx,vy,vz,8,a,b,c,d,lambda,xx,yy,zz)
         
        if (lambda .ge. 0.0d0 .and. lambda .le. 1.0d0) then
         if (iwrite .eq. 1) write(6,*) 'intersection along edges',iedge, jedge, lambda

          call check_if_vertex_exists(x_intersect,y_intersect,z_intersect,npoints,xx,yy,zz,idum)

! add this vertex to faces 2, 3, and 7
         if (idum .eq. 0) then
          npoints = npoints + 1       
          kedge(npoints) = 3
          x_intersect(npoints) = xx ; y_intersect(npoints) = yy ; z_intersect(npoints) = zz

          nvert = nvert + 1
          x_vertex(nvert) = x_intersect(npoints) ; y_vertex(nvert) = y_intersect(npoints) ; z_vertex(nvert) = z_intersect(npoints)
          ipath(nvert) = 1
          call add_vertex_to_face(2,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
          call add_vertex_to_face(3,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
          call add_vertex_to_face(7,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
          if (iwrite .eq. 1) write(6,13) 'adding vertex', nvert, x_vertex(nvert),y_vertex(nvert), z_vertex(nvert)

! if needed, add end point vertices to faces 5 and 6
          if (lambda .eq. 0.0) then
           call add_vertex_to_face(5,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
          else if (lambda .eq. 1.0) then
           call add_vertex_to_face(6,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
          end if
         end if

! end of +y lambda between 0 and 1 if block
        end if

! end of +z +y lambda between 0 and 1 if block
       end if

! end of +x lambda between 0 and 1 if block
      end if


! line intersection +x is parallel to y 
! light gray dashed line between v2 and v6, in this direction, in figure cube_paths.pdf

      iedge = 2 ; jedge = 6
      call check_edge(iedge,jedge,vx,vy,vz,8,a,b,c,d,lambda,xx,yy,zz)

      if (lambda .ge. 0.0d0 .and. lambda .le. 1.0d0) then
       if (iwrite .eq. 1) write(6,*) 'intersection along edges',iedge, jedge, lambda

       call check_if_vertex_exists(x_intersect,y_intersect,z_intersect,npoints,xx,yy,zz,idum)

! add this vertex to faces 1, 2, and 7 
       if (idum .eq. 0) then
        npoints = npoints + 1       
        kedge(npoints) = 4
        x_intersect(npoints) = xx ; y_intersect(npoints) = yy ; z_intersect(npoints) = zz

        nvert = nvert + 1
        x_vertex(nvert) = x_intersect(npoints) ; y_vertex(nvert) = y_intersect(npoints) ; z_vertex(nvert) = z_intersect(npoints)
        ipath(nvert) = 4
        call add_vertex_to_face(1,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        call add_vertex_to_face(2,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        call add_vertex_to_face(7,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        if (iwrite .eq. 1) write(6,13) 'adding vertex', nvert, x_vertex(nvert),y_vertex(nvert), z_vertex(nvert)


! if needed, add end point vertices to faces 5 and 6  
        if (lambda .eq. 0.0) then
         call add_vertex_to_face(5,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        else if (lambda .eq. 1.0) then
         call add_vertex_to_face(6,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        end if
       end if
      end if



!-------------------------------------------------------------------------------------------------------------------

! starting from vertex 1, second independent path is +y +x +z (gray solid lines in cube_paths.pdf)
! an intersection can only occur once along this path, hence the triple if block
! check +y

      iedge = 1 ; jedge = 3
      call check_edge(iedge,jedge,vx,vy,vz,8,a,b,c,d,lambda,xx,yy,zz)

      if (lambda .ge. 0.0d0 .and. lambda .le. 1.0d0) then
       if (iwrite .eq. 1) write(6,*) 'intersection along edges',iedge, jedge, lambda

       call check_if_vertex_exists(x_intersect,y_intersect,z_intersect,npoints,xx,yy,zz,idum)

! add this vertex to faces 1, 4, and 7 
       if (idum .eq. 0) then
        npoints = npoints + 1       
        kedge(npoints) = 5
        x_intersect(npoints) = xx ; y_intersect(npoints) = yy ; z_intersect(npoints) = zz

        nvert = nvert + 1
        x_vertex(nvert) = x_intersect(npoints) ; y_vertex(nvert) = y_intersect(npoints) ; z_vertex(nvert) = z_intersect(npoints)
        ipath(nvert) = 2
        call add_vertex_to_face(1,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        call add_vertex_to_face(4,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        call add_vertex_to_face(7,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        if (iwrite .eq. 1) write(6,13) 'adding vertex', nvert, x_vertex(nvert),y_vertex(nvert), z_vertex(nvert)

! if needed, add end point vertices to faces 5 and 6   
        if (lambda .eq. 0.0) then
         call add_vertex_to_face(5,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        else if (lambda .eq. 1.0) then
         call add_vertex_to_face(6,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        end if
       end if


! no intersection along +y, now check +x
! vertex v3 which contributes to faces 1, 4, and 6

      else

       nvert = nvert + 1 
       x_vertex(nvert) = vx(jedge) ; y_vertex(nvert) = vy(jedge) ; z_vertex(nvert) = vz(jedge)
       ipath(nvert) = 2
       call add_vertex_to_face(1,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
       call add_vertex_to_face(4,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
       call add_vertex_to_face(6,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
       if (iwrite .eq. 1) write(6,13) 'adding vertex', nvert, x_vertex(nvert),y_vertex(nvert), z_vertex(nvert)

       iedge = 3 ; jedge = 6
       call check_edge(iedge,jedge,vx,vy,vz,8,a,b,c,d,lambda,xx,yy,zz)
 
       if (lambda .ge. 0.0d0 .and. lambda .le. 1.0d0) then
        if (iwrite .eq. 1) write(6,*) 'intersection along edges',iedge, jedge, lambda

        call check_if_vertex_exists(x_intersect,y_intersect,z_intersect,npoints,xx,yy,zz,idum)

! add this vertex to faces 1, 6, and 7  
        if (idum .eq. 0) then
         npoints = npoints + 1       
         kedge(npoints) = 6
         x_intersect(npoints) = xx ; y_intersect(npoints) = yy ; z_intersect(npoints) = zz

         nvert = nvert + 1
         x_vertex(nvert) = x_intersect(npoints) ; y_vertex(nvert) = y_intersect(npoints) ; z_vertex(nvert) = z_intersect(npoints)
         ipath(nvert) = 2
         call add_vertex_to_face(1,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
         call add_vertex_to_face(6,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
         call add_vertex_to_face(7,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
         if (iwrite .eq. 1) write(6,13) 'adding vertex', nvert, x_vertex(nvert),y_vertex(nvert), z_vertex(nvert)

! if needed, add end point vertices to faces 4 and 2  
         if (lambda .eq. 0.0) then
          call add_vertex_to_face(4,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
         else if (lambda .eq. 1.0) then
          call add_vertex_to_face(2,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
         end if
        end if


! no intersection along +x, now check +z
! meaning we pick up vertex v6 which contributes to faces 1, 2, and  6

       else

        nvert = nvert + 1
        x_vertex(nvert) = vx(jedge) ; y_vertex(nvert) = vy(jedge) ; z_vertex(nvert) = vz(jedge)
        ipath(nvert) = 2
        call add_vertex_to_face(1,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        call add_vertex_to_face(2,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        call add_vertex_to_face(6,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        if (iwrite .eq. 1) write(6,13) 'adding vertex', nvert, x_vertex(nvert),y_vertex(nvert), z_vertex(nvert)

        iedge = 6 ; jedge = 8
        call check_edge(iedge,jedge,vx,vy,vz,8,a,b,c,d,lambda,xx,yy,zz)

        if (lambda .ge. 0.0d0 .and. lambda .le. 1.0d0) then
         if (iwrite .eq. 1) write(6,*) 'intersection along edges',iedge, jedge, lambda

         call check_if_vertex_exists(x_intersect,y_intersect,z_intersect,npoints,xx,yy,zz,idum)

! add this vertex to faces 2, 6, and 7 
         if (idum .eq. 0) then
          npoints = npoints + 1       
          kedge(npoints) = 7
          x_intersect(npoints) = xx ; y_intersect(npoints) = yy ; z_intersect(npoints) = zz
          nvert = nvert + 1
          x_vertex(nvert) = x_intersect(npoints) ; y_vertex(nvert) = y_intersect(npoints) ; z_vertex(nvert) = z_intersect(npoints)
          ipath(nvert) = 2
          call add_vertex_to_face(2,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
          call add_vertex_to_face(6,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
          call add_vertex_to_face(7,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
          if (iwrite .eq. 1) write(6,13) 'adding vertex', nvert, x_vertex(nvert),y_vertex(nvert), z_vertex(nvert)

! if needed, add end point vertices to faces 1 and 3 
          if (lambda .eq. 0.0) then
           call add_vertex_to_face(1,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
          else if (lambda .eq. 1.0) then
           call add_vertex_to_face(3,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
          end if
         end if

! end of +z lambda between 0 and 1 if block
        end if

! end of +x +z lambda between 0 and 1 if block
       end if

! end of +y +x +z lambda between 0 and 1 if block
      end if


! line intersections +y parallel to +z
! gray dashed line between v3 and v7 in figure cube_paths.pdf
      iedge = 3 ; jedge = 7
      call check_edge(iedge,jedge,vx,vy,vz,8,a,b,c,d,lambda,xx,yy,zz)
      if (lambda .ge. 0.0d0 .and. lambda .le. 1.0d0) then
       if (iwrite .eq. 1) write(6,*) 'intersection along edges',iedge, jedge, lambda

       call check_if_vertex_exists(x_intersect,y_intersect,z_intersect,npoints,xx,yy,zz,idum)

! add this vertex to faces 2, 6, and 7 
       if (idum .eq. 0) then
        npoints = npoints + 1       
        kedge(npoints) = 8
        x_intersect(npoints) = xx ; y_intersect(npoints) = yy ; z_intersect(npoints) = zz

        nvert = nvert + 1
        x_vertex(nvert) = x_intersect(npoints) ; y_vertex(nvert) = y_intersect(npoints) ; z_vertex(nvert) = z_intersect(npoints)
        ipath(nvert) = 5
        call add_vertex_to_face(4,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        call add_vertex_to_face(6,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        call add_vertex_to_face(7,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        if (iwrite .eq. 1) write(6,13) 'adding vertex', nvert, x_vertex(nvert),y_vertex(nvert), z_vertex(nvert)

! if needed, add end point vertices to faces 1 and 3   
        if (lambda .eq. 0.0) then
         call add_vertex_to_face(1,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        else if (lambda .eq. 1.0) then
         call add_vertex_to_face(3,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        end if
       end if
      end if



!-------------------------------------------------------------------------------------------------------------------     

! starting from vertex 1, third independent path is +z +y +x (black solid lines in cube_paths.pdf)
! an intersection can only occur once along this path, hence the triple if block
! check +z
      iedge = 1 ; jedge = 4
      call check_edge(iedge,jedge,vx,vy,vz,8,a,b,c,d,lambda,xx,yy,zz)

      if (lambda .ge. 0.0d0 .and. lambda .le. 1.0d0) then
       if (iwrite .eq. 1) write(6,*) 'intersection along edges',iedge, jedge, lambda

       call check_if_vertex_exists(x_intersect,y_intersect,z_intersect,npoints,xx,yy,zz,idum)

! add this vertex to faces 4, 5, and 7 
       if (idum .eq. 0) then
        npoints = npoints + 1       
        kedge(npoints) = 9
        x_intersect(npoints) = xx ; y_intersect(npoints) = yy ; z_intersect(npoints) = zz

        nvert = nvert + 1
        x_vertex(nvert) = x_intersect(npoints) ; y_vertex(nvert) = y_intersect(npoints) ; z_vertex(nvert) = z_intersect(npoints)
        ipath(nvert) = 3
        call add_vertex_to_face(4,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        call add_vertex_to_face(5,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        call add_vertex_to_face(7,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        if (iwrite .eq. 1) write(6,13) 'adding vertex', nvert, x_vertex(nvert),y_vertex(nvert), z_vertex(nvert)

! if needed, add end point vertices to faces 1 and 3
        if (lambda .eq. 0.0) then
         call add_vertex_to_face(1,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        else if (lambda .eq. 1.0) then
         call add_vertex_to_face(3,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        end if
       end if


! no intersection along +z, now check +y
! meaning we pick up vertex v4 which contributes to faces 3, 4, and 5

      else

       nvert = nvert + 1
       x_vertex(nvert) = vx(jedge) ; y_vertex(nvert) = vy(jedge) ; z_vertex(nvert) = vz(jedge)
       ipath(nvert) = 3
       call add_vertex_to_face(3,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
       call add_vertex_to_face(4,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
       call add_vertex_to_face(5,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
       if (iwrite .eq. 1) write(6,13) 'adding vertex', nvert, x_vertex(nvert),y_vertex(nvert), z_vertex(nvert)

       iedge = 4 ; jedge = 7
       call check_edge(iedge,jedge,vx,vy,vz,8,a,b,c,d,lambda,xx,yy,zz)

       if (lambda .ge. 0.0d0 .and. lambda .le. 1.0d0) then
        if (iwrite .eq. 1) write(6,*) 'intersection along edges',iedge, jedge, lambda

        call check_if_vertex_exists(x_intersect,y_intersect,z_intersect,npoints,xx,yy,zz,idum)

! add this vertex to faces 3, 4, and 7  
        if (idum .eq. 0) then
         npoints = npoints + 1       
         kedge(npoints) = 10
         x_intersect(npoints) = xx ; y_intersect(npoints) = yy ; z_intersect(npoints) = zz

         nvert = nvert + 1
         x_vertex(nvert) = x_intersect(npoints) ; y_vertex(nvert) = y_intersect(npoints) ; z_vertex(nvert) = z_intersect(npoints)
         ipath(nvert) = 3
         call add_vertex_to_face(3,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
         call add_vertex_to_face(4,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
         call add_vertex_to_face(7,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
         if (iwrite .eq. 1) write(6,13) 'adding vertex', nvert, x_vertex(nvert),y_vertex(nvert), z_vertex(nvert)

! if needed, add end point vertices to faces 5 and 6 
         if (lambda .eq. 0.0) then
          call add_vertex_to_face(5,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
         else if (lambda .eq. 1.0) then
          call add_vertex_to_face(6,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
         end if
        end if


! no intersection along +y, now check +x
! we pick up vertex v7 which contributes to faces 3, 4, and 6

       else

        nvert = nvert + 1
        x_vertex(nvert) = vx(jedge) ; y_vertex(nvert) = vy(jedge) ; z_vertex(nvert) = vz(jedge)
        ipath(nvert) = 3
        call add_vertex_to_face(3,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        call add_vertex_to_face(4,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        call add_vertex_to_face(6,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        if (iwrite .eq. 1) write(6,13) 'adding vertex', nvert, x_vertex(nvert),y_vertex(nvert), z_vertex(nvert)

        iedge = 7 ; jedge = 8
        call check_edge(iedge,jedge,vx,vy,vz,8,a,b,c,d,lambda,xx,yy,zz)

        if (lambda .ge. 0.0d0 .and. lambda .le. 1.0d0) then
         if (iwrite .eq. 1) write(6,*) 'intersection along edges',iedge, jedge, lambda

         call check_if_vertex_exists(x_intersect,y_intersect,z_intersect,npoints,xx,yy,zz,idum)

! add this vertex to faces 3, 6, and 7 
         if (idum .eq. 0) then
          npoints = npoints + 1       
          kedge(npoints) = 11
          x_intersect(npoints) = xx ; y_intersect(npoints) = yy ; z_intersect(npoints) = zz

          nvert = nvert + 1
          x_vertex(nvert) = x_intersect(npoints) ; y_vertex(nvert) = y_intersect(npoints) ; z_vertex(nvert) = z_intersect(npoints)
          ipath(nvert) = 3
          call add_vertex_to_face(3,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
          call add_vertex_to_face(6,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
          call add_vertex_to_face(7,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
          if (iwrite .eq. 1) write(6,13) 'adding vertex', nvert, x_vertex(nvert),y_vertex(nvert), z_vertex(nvert)

! if needed, add end point vertices to faces 4 and 2    
          if (lambda .eq. 0.0) then
           call add_vertex_to_face(4,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
          else if (lambda .eq. 1.0) then
           call add_vertex_to_face(2,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
          end if
         end if

! end of +x lambda between 0 and 1 if block
        end if

! end of +y +x lambda between 0 and 1 if block
       end if

! end of +z +y +x lambda between 0 and 1 if block
      end if


! line intersection +z parallel to x
! black dashed line between v4 and v5 in figure cube_paths.pdf
      iedge = 4 ; jedge = 5
      call check_edge(iedge,jedge,vx,vy,vz,8,a,b,c,d,lambda,xx,yy,zz)

      if (lambda .ge. 0.0d0 .and. lambda .le. 1.0d0) then
       if (iwrite .eq. 1) write(6,*) 'intersection along edges',iedge, jedge, lambda

       call check_if_vertex_exists(x_intersect,y_intersect,z_intersect,npoints,xx,yy,zz,idum)

! add this vertex to faces 3, 5, and 7
       if (idum .eq. 0) then
        npoints = npoints + 1       
        kedge(npoints) = 12
        x_intersect(npoints) = xx ; y_intersect(npoints) = yy ; z_intersect(npoints) = zz

        nvert = nvert + 1
        x_vertex(nvert) = x_intersect(npoints) ; y_vertex(nvert) = y_intersect(npoints) ; z_vertex(nvert) = z_intersect(npoints)
        ipath(nvert) = 6
        call add_vertex_to_face(3,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        call add_vertex_to_face(5,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        call add_vertex_to_face(7,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        if (iwrite .eq. 1) write(6,13) 'adding vertex', nvert, x_vertex(nvert),y_vertex(nvert), z_vertex(nvert)

! if needed, add end point vertices to faces 4 and 2    
        if (lambda .eq. 0.0) then
         call add_vertex_to_face(4,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        else if (lambda .eq. 1.0) then
         call add_vertex_to_face(2,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
        end if
       end if
      end if



! whew. now we have a list of potential faces and vertices
!-------------------------------------------------------------------------------------------------------------------     

      if (iwrite .eq. 2) then
       write(6,*)
       write(6,14) 'phi=',phi*rad2a, ' theta=',theta*rad2a, ' npoints =',npoints
       do n=1,npoints
        write(6,10) x_intersect(n),y_intersect(n),z_intersect(n)
       enddo
      end if
      if (iwrite .eq. 1) then
       write(6,*)
       write(6,*) 'vertices'
       do m=1,nvert
        write(6,101) m, ipath(m), x_vertex(m), y_vertex(m), z_vertex(m)
101     format(1x,2i4,1p3e12.4)
       enddo
       write(6,*)
       write(6,*) 'nfaces =',nfaces
       write(6,102) 'iface_mask =',iface_mask
102    format(1x,a,7i2)
       do m=1,nfaces
        write(6,*) face(m),nface_vertices(face(m))
        write(6,*) (ivertices(n,face(m)), n=1,nface_vertices(face(m)))
       enddo
       write(6,*)
      end if


!-------------------------------------------------------------------------------------------------------------------     

! now address various pathologies
! the utility of face_mask is thus revealed.
!
! in the degenerate case, the plane intersects one of the faces.
! remove the double counting here.
! construct a unique integer from the vertex numbers of the intersecting plane
     idum = 0
     do n=1,nface_vertices(7)
      idum = idum + 10**(n-1)*ivertices(n,7)
     end do

! construct a unique integer from the vertex numbers of the other faces
     do m= 1, nfaces
      jdum = 0
      do n=1,nface_vertices(face(m))
       jdum = jdum + 10**(n-1)*ivertices(n,face(m))
      enddo

! zero the face_mask if the two integers match
      if (jdum .eq. idum .and. face(m) .ne. 7) then
       iface_mask(face(m)) = 0
       if (iwrite .eq. 1) then
        write(6,*) 'degenerate face'
        write(6,*) 'removing face ',m,face(m)
        write(6,*) (ivertices(n,7), n=1,nface_vertices(7))
        write(6,*) (ivertices(n,face(m)), n=1,nface_vertices(face(m)))
       end if
      end if

! another case to address is no intersection at all and only cube vertices.
! remove the faces where the number of vertices is less than 3
      if (iface_mask(face(m)) .eq. 1 .and. nface_vertices(face(m)) .lt. 3) then
       iface_mask(face(m)) = 0
       if (iwrite .eq. 1) then
        write(6,*) 'face with less than 3 vertices'
        write(6,*) 'removing face ',m,face(m),nface_vertices(face(m))
       end if
      end if
     end do


!-------------------------------------------------------------------------------------------------------------------     


! compute the volume & area
! if we got any intersection

     if (npoints .gt. 0) then
      volume = 0.0d0
      area_frac = 0.0d0    ! added Hansinger

! loop over valid faces
      do m = 4,nfaces
       if (iface_mask(face(m)) .eq. 1) then


! the vertices of the intersecting plane, by construction, are in clockwise order.
! the cube face intersections do not have a consistent winding; 
! we put them in clockwise order by reversing paths 

! face 1
! reverse order of vertexes along path 2, dark gray in cube_paths.pdf

       if (face(m) .eq. 1) then
        idum = 0
        do n = 1, nface_vertices(face(m))
         kdum = ivertices(n,face(m))   
         if (ipath(kdum) .eq. 0 .or. ipath(kdum) .eq. 1 .or. ipath(kdum) .eq. 4) then
          idum = idum + 1
          xdum(idum) = x_vertex(kdum) ; ydum(idum) = y_vertex(kdum) ; zdum(idum) = z_vertex(kdum)
         end if
        end do
        do n = nface_vertices(face(m)), 1, -1
         kdum = ivertices(n,face(m)) 
         if (ipath(kdum) .eq. 2) then
          idum = idum + 1
          xdum(idum) = x_vertex(kdum) ; ydum(idum) = y_vertex(kdum) ; zdum(idum) = z_vertex(kdum)
         end if
        end do

        if (idebug .eq. 1) then
         call irregular_polygon_area(xdum,ydum,zdum,nface_vertices(face(m)),area)
         if (area .eq. 0.0) then
          write(6,14) 'dl =',dl, ' phi  =',phi*rad2a, ' theta=',theta*rad2a
          write(6,11) (n,xdum(n),ydum(n),zdum(n), n=1,nface_vertices(face(m)))
          write(6,*) 'zero area face 1 - bad ordering'
         end if
        end if


! face 2
! reverse order of vertexes along paths 2 and 4, dark gray and dotted light gray in cube_paths.pdf

       else if (face(m) .eq. 2) then
        idum = 0
        do n = 1, nface_vertices(face(m))
         kdum = ivertices(n,face(m))
         if (ipath(kdum) .eq. 1 ) then
          idum = idum + 1
          xdum(idum) = x_vertex(kdum) ; ydum(idum) = y_vertex(kdum) ; zdum(idum) = z_vertex(kdum)
         end if
        end do
        do n = nface_vertices(face(m)), 1, -1
         kdum = ivertices(n,face(m))
         if (ipath(kdum) .eq. 2 .or. ipath(kdum) .eq. 4 ) then
          idum = idum + 1
          xdum(idum) = x_vertex(kdum) ; ydum(idum) = y_vertex(kdum) ; zdum(idum) = z_vertex(kdum)
         end if
        end do
        if (idebug .eq. 1) then
         call irregular_polygon_area(xdum,ydum,zdum,nface_vertices(face(m)),area)
         if (area .eq. 0.0) then
          write(6,14) 'dl =',dl, ' phi  =',phi*rad2a, ' theta=',theta*rad2a
          write(6,11) (n,xdum(n),ydum(n),zdum(n), n=1,nface_vertices(face(m)))
          write(6,*) 'zero area face 2 - bad ordering'
         end if
        end if


! face 3
! reverse order of vertexes along path 3 , black in cube_paths.pdf 

       else if (face(m) .eq. 3) then
        idum = 0
        do n = 1, nface_vertices(face(m))
         kdum = ivertices(n,face(m))
         if (ipath(kdum) .eq. 1 .or. ipath(kdum) .eq. 6 ) then
          idum = idum + 1
          xdum(idum) = x_vertex(kdum) ; ydum(idum) = y_vertex(kdum) ; zdum(idum) = z_vertex(kdum)
         end if
        end do
        do n = nface_vertices(face(m)), 1, -1
         kdum = ivertices(n,face(m))
         if (ipath(kdum) .eq. 3 ) then
          idum = idum + 1
          xdum(idum) = x_vertex(kdum) ; ydum(idum) = y_vertex(kdum) ; zdum(idum) = z_vertex(kdum)
         end if
        end do
        if (idebug .eq. 1) then
         call irregular_polygon_area(xdum,ydum,zdum,nface_vertices(face(m)),area)
         if (area .eq. 0.0) then
          write(6,14) 'dl =',dl, ' phi  =',phi*rad2a, ' theta=',theta*rad2a
          write(6,11) (n,xdum(n),ydum(n),zdum(n), n=1,nface_vertices(face(m)))
          write(6,*) 'zero area face 3'
         end if
        end if


! face 4
! reverse order of vertexes along path 3 , black in cube_paths.pdf 

       else if (face(m) .eq. 4) then
        idum = 0
        do n = 1, nface_vertices(face(m))
         kdum = ivertices(n,face(m))
         if (ipath(kdum) .eq. 0 .or. ipath(kdum) .eq. 2 .or. ipath(kdum) .eq. 5 ) then
          idum = idum + 1
          xdum(idum) = x_vertex(kdum) ; ydum(idum) = y_vertex(kdum) ; zdum(idum) = z_vertex(kdum)
         end if
        end do
        do n = nface_vertices(face(m)), 1, -1
         kdum = ivertices(n,face(m))
         if (ipath(kdum) .eq. 3 ) then
          idum = idum + 1
          xdum(idum) = x_vertex(kdum) ; ydum(idum) = y_vertex(kdum) ; zdum(idum) = z_vertex(kdum)
         end if
        end do
        if (idebug .eq. 1) then
         call irregular_polygon_area(xdum,ydum,zdum,nface_vertices(face(m)),area)
         if (area .eq. 0.0) then
          write(6,14) 'dl =',dl, ' phi  =',phi*rad2a, ' theta=',theta*rad2a
          write(6,11) (n,xdum(n),ydum(n),zdum(n), n=1,nface_vertices(face(m)))
          write(6,*) 'zero area face 4'
         end if
        end if


! face 5
! reverse order of vertexes along paths 3 and 6, black and dark dashed in cube_paths.pdf 

       else if (face(m) .eq. 5) then
        idum = 0
        do n = 1, nface_vertices(face(m))
         kdum = ivertices(n,face(m))
         if (ipath(kdum) .eq. 0 .or. ipath(kdum) .eq. 1 ) then
          idum = idum + 1
          xdum(idum) = x_vertex(kdum) ; ydum(idum) = y_vertex(kdum) ; zdum(idum) = z_vertex(kdum)
         end if
        end do
        do n = nface_vertices(face(m)), 1, -1
         kdum = ivertices(n,face(m))
         if (ipath(kdum) .eq. 3 .or. ipath(kdum) .eq. 6 ) then
          idum = idum + 1
          xdum(idum) = x_vertex(kdum) ; ydum(idum) = y_vertex(kdum) ; zdum(idum) = z_vertex(kdum)
         end if
        end do
        if (idebug .eq. 1) then
         call irregular_polygon_area(xdum,ydum,zdum,nface_vertices(face(m)),area)
         if (area .eq. 0.0) then
          write(6,14) 'dl =',dl, ' phi  =',phi*rad2a, ' theta=',theta*rad2a
          write(6,11) (n,xdum(n),ydum(n),zdum(n), n=1,nface_vertices(face(m)))
          write(6,*) 'zero area face 5'
         end if
        end if


! face 6
! reverse order of vertexes along paths 3 and 5, black and gray in cube_paths.pdf 

       else if (face(m) .eq. 6) then
        idum = 0
        do n = 1, nface_vertices(face(m))
         kdum = ivertices(n,face(m))
         if (ipath(kdum) .eq. 2 ) then
          idum = idum + 1
          xdum(idum) = x_vertex(kdum) ; ydum(idum) = y_vertex(kdum) ; zdum(idum) = z_vertex(kdum)
         end if
        end do
        do n = nface_vertices(face(m)), 1, -1
         kdum = ivertices(n,face(m))
         if (ipath(kdum) .eq. 3 .or. ipath(kdum) .eq. 5 ) then
          idum = idum + 1
          xdum(idum) = x_vertex(kdum) ; ydum(idum) = y_vertex(kdum) ; zdum(idum) = z_vertex(kdum)
         end if
        end do
        if (idebug .eq. 1) then
         call irregular_polygon_area(xdum,ydum,zdum,nface_vertices(face(m)),area)
         if (area .eq. 0.0) then
          write(6,14) 'dl =',dl, ' phi  =',phi*rad2a, ' theta=',theta*rad2a
          write(6,11) (n,xdum(n),ydum(n),zdum(n), n=1,nface_vertices(face(m)))
          write(6,*) 'zero area face 6'
         end if
        end if


! face 7 is the intersecting plane and the final face
! they are in clockwise order by construction

       else if (face(m) .eq. 7) then
        do n = 1, nface_vertices(face(m))
         kdum = ivertices(n,face(m))
         xdum(n) = x_vertex(kdum) ; ydum(n) = y_vertex(kdum) ; zdum(n) = z_vertex(kdum)
        end do
       end if


! volume of this polyhedra
! choose vertex 1 as the reference height of the pyramids.
! faces 1, 4, and 5 then contribute zero volume because their heights are zero.
! only faces 2, 3, 6 and 7 contribute.

        call pyramid_volume(xdum,ydum,zdum,nface_vertices(face(m)),x_vertex(1),y_vertex(1),z_vertex(1),dv)
        call pyramid_area(xdum,ydum,zdum,nface_vertices(face(m)),x_vertex(1),y_vertex(1),z_vertex(1),area) ! added Hansinger

        volume = volume + dv
        area_frac = area   ! added Hansinger

        if (iwrite .eq. 1) then
         write(6,12) m,face(m),volume, dv
        else if (iwrite .eq. 2) then
         write(6,*) ' face ', face(m),'  nvertices', nface_vertices(face(m)) 
         write(6,'(1pe12.4,1pe12.4,1pe12.4)') (xdum(n),ydum(n),zdum(n), n = 1, nface_vertices(face(m)) )
         write(6,'(a,1pe12.4)') 'dv ',dv
        end if


! end of loop over valid faces
       end if
      enddo


! check the volume is between 0.0 and 1.0
      if (iwrite .eq. 1) write(6,*) 'volume =',volume

      if (volume .le. 0.0d0) then
       write(6,*)
       write(6,*) 'volume is negative',volume
       write(6,10) theta*rad2a, phi*rad2a, dl, volume
       write(6,*) npoints
       write(6,12) (n, kedge(n), x_intersect(n), y_intersect(n), z_intersect(n), n=1,npoints)
!       stop 'bad volume'
      end if

      if (volume .gt. 1.0d0) then
       write(6,*)
       write(6,*) 'volume of unit cube > 1 ',volume
       write(6,10) theta*rad2a, phi*rad2a, dl, volume
       write(6,*) npoints
       write(6,12) (n, kedge(n), x_intersect(n), y_intersect(n), z_intersect(n), n=1,npoints)
!       stop 'bad volume'
      end if

! end of npoints if block on the volume calculation
      end if

      return
      end subroutine area_fraction
      
! End edit Hansinger
! #######################



      subroutine add_vertex_to_face(iface,nfaces,iface_mask,face,nface_vertices,nvert,ivertices)
      implicit none

! adds a face for the plane - cube intersection

! declare the pass
      integer :: iface,nfaces,iface_mask(7),face(7),nface_vertices(7),nvert,ivertices(7,7)

! go
      if (iface_mask(iface) .eq. 0) then
       iface_mask(iface) = 1
       nfaces = nfaces + 1
       face(nfaces) = iface
      end if
      nface_vertices(iface) = nface_vertices(iface) + 1
      ivertices(nface_vertices(iface),iface) = nvert            
      return
      end


      subroutine check_if_vertex_exists(x_intersect,y_intersect,z_intersect,n,xx,yy,zz,idum)
      implicit none

! checks if a vertex already exists
      
! declare the pass
      integer    :: n, idum
      real*8     :: x_intersect(n),y_intersect(n),z_intersect(n),xx,yy,zz

! local variables
      integer           :: i
      real*8, parameter :: tiny = 1.0d-12

! loop over all existing intersection points
      idum = 0
      do i=1,n
       if (abs(x_intersect(i)-xx) .le. tiny .and. & 
           abs(y_intersect(i)-yy) .le. tiny .and. & 
           abs(z_intersect(i)-zz) .le. tiny) then
        idum = 1
        exit
       end if
      enddo

      return
      end subroutine check_if_vertex_exists





      subroutine check_edge(iedge,jedge,vx,vy,vz,n,a,b,c,d, & 
                            lambda,xx,yy,zz)
      implicit none

! computes the distance lambda along an edge between two vertices where an intersection occurs.
! lambda will be between 0 and 1 in the unit cube for a valid intersection

! input: 
! iedge = edge number
! jedge = edge numver
! vx, vy, vz = coordinates of unit cube
! a, b, c, d = coefficients of the hessian normal plane ax + b*y * c*z + d = 0

! output:
! lambda = distance along edges for an intersection
! xx, yy, zz = coordinates of intersection point

! declare the pass
      integer    :: iedge, jedge, n
      real*8     :: vx(n), vy(n), vz(n), a, b, c, d, lambda, xx, yy, zz

! local variables
      real*8            :: eij(3)
      real*8, parameter :: tiny = 1.0d-12

      eij(1) = vx(jedge) - vx(iedge) ; eij(2) = vy(jedge) - vy(iedge) ;  eij(3) = vz(jedge) - vz(iedge)

      lambda = (d - (a*vx(iedge) + b*vy(iedge) + c*vz(iedge)) ) / (a*eij(1) + b*eij(2) + c*eij(3))

!      write(6,'(a,1p2e24.16)') 'check edge ',lambda, abs(lambda - 1.0d0)

      if (abs(lambda) .lt. tiny) lambda = 0.0d0
      if (abs(lambda - 1.0d0) .lt. tiny) lambda = 1.0d0

! coordinates of intersection 
      xx = 0.0d0 ; yy = 0.0d0 ; zz = 0.0d0
      if (lambda .ge. 0.0d0 .and. lambda .le. 1.0d0) then
       xx = vx(iedge)+lambda*eij(1) ; yy = vy(iedge)+lambda*eij(2) ; zz = vz(iedge)+lambda*eij(3)
      end if

      return
      end subroutine check_edge






      include 'polylib.f90'
