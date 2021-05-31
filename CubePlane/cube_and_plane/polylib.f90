! a collection of routine for polygons (2d) and polyhedra (3d)

! routine irregular_polygon_area gets the area and centroid 
! routine polygon_cross_product does a x b
! routine get_polygon_normal creates a polygon's normal unit vector
! routine get_polygon_local_coord converts a polygon in 3d (x,y,z) to a polygon in 2d (x',y',0)
! routine pyramid_height get the height of a pyramid from the base to a top point  
! routine pyramid_volume gets the volume of a pyramid




      subroutine irregular_polygon_area(x,y,z,n,area)
      implicit none

! given the coordinates of the vertices, this routine applies 
! the surveyor's formula for the area and centroid of an irregular 2d polygon
! the vertices may be ordered clockwise or counterclockwise. if they are 
! ordered clockwise, the area will be negative but correct in absolute value.

! it is assumed the vertices are ordered in a counterclockwise manner.
!
! x(n)    = x-coordinates of vertices
! y(n)    = y-coordinates of vertices
! z(n)    = z-coordinates of vertices
! n       = number of vertices
! area    = area of irregular polygon


! declare the pass
      integer            :: n
      real*8             ::  x(n),y(n),z(n),area

! local variables
      integer            :: i
      integer, parameter :: nmax = 200  ! maximum  number of vertices
      real*8             :: xp(nmax),yp(nmax),zp(nmax),det,sum1,sum2,sum3
      real*8             :: xcenter, ycenter
      real*8, parameter  :: sixth = 1.0d0/6.0d0

! check
      if (n .gt. nmax) stop 'n > nmax in routine irregular_polygon_area'


! put the polygon in its own x-y coordinates; all zp's should be zero

      call get_polygon_local_coord(x,y,z,n,xp,yp,zp)


! apply the surveyor's formula
      sum1 = 0.0d0
      sum2 = 0.0d0
      sum3 = 0.0d0
      do i=1,n-1
       det  = xp(i)*yp(i+1) - xp(i+1)*yp(i)
       sum1 = sum1 + det
       sum2 = sum2 + det*(xp(i) + xp(i+1))
       sum3 = sum3 + det*(yp(i) + yp(i+1))
      end do

! close the polygon
      det  = xp(n)*yp(1) - xp(1)*yp(n)
      sum1 = sum1 + det
      sum2 = sum2 + det*(xp(n) + xp(1))
      sum3 = sum3 + det*(yp(n) + yp(1))

! area and center
      area    = 0.5d0 * sum1
      det     = 1.0d0/area
      xcenter = sixth*det * sum2
      ycenter = sixth*det * sum3
      area    = abs(area)

      return
      end subroutine irregular_polygon_area




      subroutine polygon_cross_product(a,b,c)
      implicit none
      real*8 :: a(3),b(3),c(3)
      c(1) = a(2)*b(3) - b(2)*a(3)
      c(2) = a(3)*b(1) - b(3)*a(1)
      c(3) = a(1)*b(2) - b(1)*a(2)
      return
      end subroutine polygon_cross_product





      subroutine get_polygon_normal(v1,v2,v3,norm)
      implicit none

! produce the unit vector normal to the polygon

! declare the pass
      real*8            :: v1(3),v2(3),v3(3),norm(3)

! local variables
      real*8            :: a(3),b(3),mag
      real*8, parameter :: tiny=1.0d-14

      a = v2 - v1
      b = v3 - v2
      call polygon_cross_product(a,b,norm)
      mag  = max(sqrt(dot_product(norm,norm)), tiny)
      norm = norm/mag
      return
      end subroutine get_polygon_normal





      subroutine get_polygon_local_coord(x,y,z,n,xprime,yprime,zprime)
      implicit none
 
! convert a polygon in (x,y,z) to a polygon in (x',y',0)

! input
! x(n)    = x-coordinates of polygon vertices
! y(n)    = y-coordinates of polygon vertices
! z(n)    = z-coordinates of polygon vertices
! n       = number of vertices

! output
! xprime(n) = x-coordinates of polygon vertices
! yprime(n) = y-coordinates of polygon vertices
! zprime(n) = z-coordinates of polygon vertices


! declare the pass
      integer       :: n
      real*8        :: x(n),y(n),z(n),xprime(n),yprime(n),zprime(n)

! local variables
      integer            :: i
      integer, parameter :: nmax = 200   ! maximum number of vertices
      real*8             :: xc(nmax),yc(nmax),zc(nmax), &
                            iprime(3),jprime(3),kprime(3),pv(3),mag

! copy the input
      if (n .gt. nmax) stop 'n > nmax in routine get_polygon_local_coord'

      xc(1:n) = x(1:n) ; yc(1:n) = y(1:n) ; zc(1:n) = z(1:n)

! subtract the displacement from all points
      xc(1:n) = xc(1:n) - xc(1)
      yc(1:n) = yc(1:n) - yc(1)
      zc(1:n) = zc(1:n) - zc(1)


! select point 1 to point 2 as the x direction
! unit vector in the xprime direction
  
      iprime= (/xc(2) - xc(1), yc(2) - yc(1), zc(2) - zc(1)/)
      mag = max(sqrt(dot_product(iprime,iprime)), 1.0d-14)
      iprime = iprime/mag


! vector from point 2 to point 3
      pv = (/xc(3) - xc(2), yc(3) - yc(2), zc(3) - zc(2)/)

! unit vector in the zprime direction
      call polygon_cross_product(iprime,pv,kprime)
      mag = max(sqrt(dot_product(kprime,kprime)), 1.0d-14)
      kprime = kprime/mag


! unit vector jprime in the yprime direction
      call polygon_cross_product(kprime,iprime,jprime)
      mag = max(sqrt(dot_product(jprime,jprime)), 1.0d-14)
      jprime = jprime/mag


! for each point find the projections of xprime, yprime, and zprime
! all zprime values should be zero
      do i=1,n
       pv        = (/xc(i),yc(i),zc(i)/)
       xprime(i) = dot_product(iprime,pv)
       yprime(i) = dot_product(jprime,pv)
       zprime(i) = dot_product(kprime,pv)
      enddo
      return
      end subroutine get_polygon_local_coord




      subroutine pyramid_height(x,y,z,n,px,py,pz,dist)
      implicit none

! get the height of the pyramid from the base to the top point

! input
! x(n)    = x-coordinates of polygon vertices
! y(n)    = y-coordinates of polygon vertices
! z(n)    = z-coordinates of polygon vertices
! n       = number of vertices
! px      = x-coordinate of point
! py      = x-coordinate of point
! px      = x-coordinate of point

! output
! dist    = distance from base to point

! declare the pass
      integer  :: n
      real*8   :: x(n),y(n),z(n),px,py,pz,dist

! local variables
      real*8   :: v1(3),v2(3),v3(3),norm(3),vt(3)


! get unit vector normal to polygon 
      v1 = (/x(1),y(1),z(1)/)
      v2 = (/x(2),y(2),z(2)/)
      v3 = (/x(3),y(3),z(3)/)
      call get_polygon_normal(v1,v2,v3,norm)


! the height is the projection of vt in the normal direction.
! absolute value to be independent of orientation 

      vt = (/px-x(1), py-y(1), pz-z(1)/)
      dist = abs(dot_product(norm,vt))
      return
      end subroutine pyramid_height




      subroutine pyramid_volume(x,y,z,n,px,py,pz,volume)
      implicit none

! volume of a pyramid

! input
! x(n)    = x-coordinates of polygon vertices
! y(n)    = y-coordinates of polygon vertices
! z(n)    = z-coordinates of polygon vertices
! n       = number of vertices
! px      = x-coordinate of point
! py      = x-coordinate of point
! px      = x-coordinate of point

! output
! volume = volume of pyramid

! declare the pass
      integer           :: n
      real*8            :: x(n),y(n),z(n),px,py,pz,volume

! local variables
      integer           :: i
      real*8            :: area,height
      real*8, parameter :: third = 1.0d0/3.0d0


      call irregular_polygon_area(x,y,z,n,area)
      call pyramid_height(x,y,z,n,px,py,pz,height)
      volume = area * height * third

!      write(6,'(a,1pe9.2,a,1pe9.2,a,1pe9.2)') 'area   ', area,'  height ', height,'  volume ', volume

      return
      end subroutine pyramid_volume
      
! ######################
! added Hansinger
      subroutine pyramid_area(x,y,z,n,px,py,pz,this_area)
      implicit none

! volume of a pyramid

! input
! x(n)    = x-coordinates of polygon vertices
! y(n)    = y-coordinates of polygon vertices
! z(n)    = z-coordinates of polygon vertices
! n       = number of vertices
! px      = x-coordinate of point
! py      = x-coordinate of point
! px      = x-coordinate of point

! output
! volume = volume of pyramid

! declare the pass
      integer           :: n
      real*8            :: x(n),y(n),z(n),px,py,pz,this_area

! local variables
      integer           :: i
      real*8            :: area,height
      real*8, parameter :: third = 1.0d0/3.0d0


      call irregular_polygon_area(x,y,z,n,area)
      !call pyramid_height(x,y,z,n,px,py,pz,height)
      !volume = area * height * third
      this_area=area

!      write(6,'(a,1pe9.2,a,1pe9.2,a,1pe9.2)') 'area   ', area,'  height ', height,'  volume ', volume

      return
      end subroutine pyramid_area
