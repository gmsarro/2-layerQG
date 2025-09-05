!Created by Noboru Nakamura. Edited by Joonsuk M. Kang and Giorgio M. Sarro.
!****** Read .nc file for the upper layer QGPV and Qref *****
!****** and compute LWA *****
!Edit the model dimensions!

        program PV

        use NETCDF

        integer, parameter:: imax = 128,jmax=128
        integer, parameter:: n = 10000
        common /array/ qgpv1(imax,jmax,n)
        common /brray/ qref1(jmax,n)
        common /crray/ qe(imax,jmax)
        common /drray/ waa1(imax,jmax,n)
        common /erray/ wac1(imax,jmax,n)
        common /frray/ wa1(imax,jmax,n)
        integer :: ncid, ncid2,stat,nDim,nVar,nAtt,uDimID,inq
        integer :: lonID,latID,vid2,varID
        integer :: l1,l2,l3,l4,l5,l6,xtype,len,attnum

! ### Model dimensions ###
        wx = 72.   ! 28000 km x 28000 km
        wy = 96.   ! 28000 km x 28000 km
        dx = wx/128.
        dy = wy/127.
        
! ### Read the .nc file for PV ####
       stat = nf90_open('QGPV.nc',nf90_nowrite,ncid)
       stat = nf90_inquire(ncid,nDim,nVar,nAtt,uDimID)
       write(6,*) 'ndim,nvar,natt,uDimID =',nDim,nVar,nAtt,uDimID
       stat = nf90_inq_varid(ncid,"q1",varID)
       write(6,*) 'Variable ID for qgpv1 = ',varID
       stat = nf90_get_var(ncid,varID,qgpv1)
       stat = nf90_close(ncid)  
       

! ### Read the .nc file for Qref ### 
       stat = nf90_open('qref1.nc',nf90_nowrite,ncid)
       stat = nf90_inquire(ncid,nDim,nVar,nAtt,uDimID)
       write(6,*) 'ndim,nvar,natt,uDimID =',nDim,nVar,nAtt,uDimID
       stat = nf90_inq_varid(ncid,"qref1",varID)
       write(6,*) 'Variable ID for qref1 = ',varID
       stat = nf90_get_var(ncid,varID,qref1)
       stat = nf90_close(ncid)

! ### Time loop ###
       do m = 1,n
       do jjj = 1, jmax 
      ! ### Compute qe ###
         qe(:,:) = qgpv1(:,:,m)-qref1(jjj,m) 

        do i = 1,imax  !2
         waa1(i,jjj,m) = 0.
         wac1(i,jjj,m) = 0.
         do j = 1,jmax  !3
           if(j.lt.jjj.and.qe(i,j).gt.0.) then
             wac1(i,jjj,m)=wac1(i,jjj,m)+qe(i,j)*dy
           endif
           if(j.ge.jjj.and.qe(i,j).le.0.) then
             waa1(i,jjj,m)=waa1(i,jjj,m)-qe(i,j)*dy
           endif
         enddo !3
        enddo !2
        enddo !jjj loop
        write(6,*) 'normal end =',m
       end do

! ### Create .nc file for waa1 ###
       stat = nf90_create('waa1.nc',nf90_noclobber,ncid2)
       stat = nf90_def_dim(ncid2,"time",10000,it)
       stat = nf90_def_dim(ncid2,"latitude",128,iy) !akc
       stat = nf90_def_dim(ncid2,"longitude",128,ix)
       stat = nf90_def_var(ncid2,"waa1",nf90_float,   &
                (/ix,iy,it/), vid2)
       stat = nf90_put_att(ncid2,vid2,"title",'waa1.nc')
       stat = nf90_enddef(ncid2)
       stat = nf90_put_var(ncid2,vid2,waa1)
       stat = nf90_close(ncid2)

! ### File consistency check ####
       stat = nf90_open('waa1.nc',nf90_nowrite,ncid)
       stat = nf90_inquire(ncid,nDim,nVar,nAtt,uDimID)
       write(6,*) 'ndim,nvar,natt,uDimID =',nDim,nVar,nAtt,uDimID
       stat = nf90_inq_varid(ncid,"waa1",varID)
       write(6,*) 'Variable ID for waa1 = ',varID
       stat = nf90_get_var(ncid,varID,wa1)
       stat = nf90_close(ncid)

       write(6,*) 'Data consistency: ',waa1(64,64,125), wa1(64,64,125)

! ### Create .nc file for wac1 ###
       stat = nf90_create('wac1.nc',nf90_noclobber,ncid2)
       stat = nf90_def_dim(ncid2,"time",10000,it)
       stat = nf90_def_dim(ncid2,"latitude",128,iy)!akc
       stat = nf90_def_dim(ncid2,"longitude",128,ix)
       stat = nf90_def_var(ncid2,"wac1",nf90_float,   &
                (/ix,iy,it/), vid2)!akc
       stat = nf90_put_att(ncid2,vid2,"title",'wac1.nc')
       stat = nf90_enddef(ncid2)
       stat = nf90_put_var(ncid2,vid2,wac1)
       stat = nf90_close(ncid2)

! ### File consistency check ####
       stat = nf90_open('wac1.nc',nf90_nowrite,ncid)
       stat = nf90_inquire(ncid,nDim,nVar,nAtt,uDimID)
       write(6,*) 'ndim,nvar,natt,uDimID =',nDim,nVar,nAtt,uDimID
       stat = nf90_inq_varid(ncid,"wac1",varID)
       write(6,*) 'Variable ID for wac1 = ',varID
       stat = nf90_get_var(ncid,varID,wa1)
       stat = nf90_close(ncid)

       write(6,*) 'Data consistency: ',wac1(18,18,10), wa1(18,18,10)

      stop
    end program
