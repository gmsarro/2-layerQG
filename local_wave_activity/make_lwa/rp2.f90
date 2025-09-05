!Created by Noboru Nakamura. Edited by Joonsuk M. Kang and Giorgio M. Sarro.
!****** Read .nc file for the upper layer QGPV *****
!****** and compute Qref *****
!Edit the model dimensions!
program PV

      use NETCDF

      integer, parameter:: imax = 128,jmax=128
      integer, parameter:: n = 10000
      common /array/ qgpv1(imax,jmax,n)
      common /brray/ qn(jmax),an(jmax),cn(jmax),qref(jmax,n)
      common /crray/ q1(imax,jmax)
      common /drray/ wa1(jmax,n)
      real :: cz(jmax)
      real :: C, Er, wx, wy, dx, dy, qmin, qmax, dq, d1
      integer :: ncid, ncid2,stat,nDim,nVar,nAtt,uDimID,inq
      integer :: lonID,latID,vid2,varID
      integer :: l1,l2,l3,l4,l5,l6,xtype,len,attnum

      character(len=100) :: filename
      character(len=100) :: filecreate
      
      real :: L_arr(11), U_arr(22)
      integer :: io, jo
      character(len=6) :: L_number
      character(len=6) :: U_number

        C = 2.0
        Er = 0.1        
! ### Model dimensions ###
        wx = 72.   ! 28000 km x 28000 km
        wy = 96.   ! 28000 km x 28000 km
        dx = wx/128.
        dy = wy/127.

! ### Read the .nc file ####
       stat = nf90_open('QGPV.nc',nf90_nowrite,ncid) 
       stat = nf90_inquire(ncid,nDim,nVar,nAtt,uDimID)
       write(6,*) 'ndim,nvar,natt,uDimID =',nDim,nVar,nAtt,uDimID
       stat = nf90_inq_varid(ncid,"q1",varID)
       write(6,*) 'Variable ID for qgpv1 = ',varID
       stat = nf90_get_var(ncid,varID,qgpv1)
       stat = nf90_close(ncid)  

! ### Time loop ###
       do m = 1,n
        q1(:,:) = qgpv1(:,:,m)

    ! ### Min & Max values and bin size for PV ###     
         qmin = minval(q1)
         qmax = maxval(q1)
         dq = (qmax-qmin)/127.  ! 257 bins with equal size 

    ! ### Create PV bins and area bins ###
         do j = 1,jmax
          qn(j) = qmin+float(j-1)*dq
          cz(j) = float(j-1)*dy*wx ! (J.Kang wx instead of wl)
         enddo

    ! ### Tally area according to PV values ###
          an(:) = 0.
        do j = 1,jmax
        do i = 1,imax  !2
          k = int((q1(i,j)-qmin)/dq)+1
          an(k) = an(k)+dx*dy
        enddo !2
        enddo
          cn(1) = 0.
        
        do j = 2,jmax
          cn(j) = cn(j-1)+an(j-1)
        enddo
        cn(jmax) = cn(jmax)+an(jmax)

     ! ### Interpolate for Qref ###
          qref(:,m) = 100.
          qref(1,m) = qmin
          qref(jmax,m) = qmax
        do j = 2,jmax-1
          do jj = 1,jmax-1
           if((cz(j).ge.cn(jj)).and.(cz(j).lt.cn(jj+1))) then
             d1 = (cz(j)-cn(jj))/(cn(jj+1)-cn(jj))
             qref(j,m) = d1*qn(jj+1)+(1.-d1)*qn(jj)
           endif
          enddo
        enddo

        write(6,*) 'normal end =',m
       end do

       stat = nf90_create('qref1.nc',nf90_noclobber,ncid2)

       stat = nf90_def_dim(ncid2,"time",10000,it)
       stat = nf90_def_dim(ncid2,"latitude",128,iy)
       stat = nf90_def_var(ncid2,"qref1",nf90_float,   &
                (/iy,it/), vid2)
       stat = nf90_put_att(ncid2,vid2,"title",'qref1.nc')
       stat = nf90_enddef(ncid2)
       stat = nf90_put_var(ncid2,vid2,qref)
       stat = nf90_close(ncid2)

! ### File consistency check ####
       stat = nf90_open('qref1.nc',nf90_nowrite,ncid)
       stat = nf90_inquire(ncid,nDim,nVar,nAtt,uDimID)
       write(6,*) 'ndim,nvar,natt,uDimID =',nDim,nVar,nAtt,uDimID
       stat = nf90_inq_varid(ncid,"qref1",varID)
       write(6,*) 'Variable ID for qref1 = ',varID
       stat = nf90_get_var(ncid,varID,wa1)
       stat = nf90_close(ncid)

       write(6,*) 'Data consistency: ',qref(64,5000), wa1(64,5000)
      stop
end program
