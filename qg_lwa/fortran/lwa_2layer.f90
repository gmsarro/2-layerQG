! Created by Noboru Nakamura. Edited by Joonsuk M. Kang, Giorgio M. Sarro, and S. Smith
! ****** compute Qref and LWA for 2 layer model *****
! build with below (asterisk to avoid accidental f2py processing)
! *f2py -c --opt="-fbounds-check -Wextra" -m lwa_2layer lwa_2layer.f90
module lwa_2layer
  implicit none
  private
  public :: calc_qref, calc_lwa

  contains


  subroutine calc_qref(qgpv1, imax, jmax, n, wx, wy, qref, verbose)
    implicit none
    ! ### Model dimensions ###
    ! f2py intent(hide) :: imax, jmax, n
    integer, intent(in) :: imax, jmax, n
    real, intent(in) :: qgpv1(imax,jmax,n), wx, wy
    real, intent(out) :: qref(jmax, n)
    logical, optional :: verbose
    real :: qn(jmax), an(jmax), cn(jmax), cz(jmax)
    real :: q1(imax, jmax)
    real :: dx, dy, qmin, qmax, dq, d1
    integer :: i, j, jj, k, m

    if (.not. present(verbose)) verbose = .FALSE.

    dx = wx/real(imax)
    dy = wy/real(jmax - 1)

  ! ### Time loop ###
    do m = 1, n
      q1(:, :) = qgpv1(:, :, m)

      ! ### Min & Max values and bin size for PV ###
      qmin = minval(q1)
      qmax = maxval(q1)
      dq = (qmax - qmin)/float(jmax - 1)  ! 257 bins with equal size

      ! ### Create PV bins and area bins ###
      do j = 1, jmax
        qn(j) = qmin + float(j - 1)*dq
        cz(j) = float(j - 1)*dy*wx ! (J.Kang wx instead of wl)
      end do

      ! ### Tally area according to PV values ###
      an(:) = 0.
      do j = 1, jmax
        do i = 1, imax  !2
          k = int((q1(i, j) - qmin)/dq) + 1
          an(k) = an(k) + dx*dy
        end do !2
      end do
      cn(1) = 0.
      do j = 2, jmax
        ! cn(j) = cn(j-1)+0.5*(an(j)+an(j-1))
        cn(j) = cn(j - 1) + an(j - 1)
      end do
      cn(jmax) = cn(jmax) + an(jmax)

      ! ### Interpolate for Qref ###
      qref(:, m) = 100.
      qref(1, m) = qmin
      qref(jmax, m) = qmax
      do j = 2, jmax - 1
        do jj = 1, jmax - 1
          if ((cz(j) .ge. cn(jj)) .and. (cz(j) .lt. cn(jj + 1))) then
            d1 = (cz(j) - cn(jj))/(cn(jj + 1) - cn(jj))
            qref(j, m) = d1*qn(jj + 1) + (1.-d1)*qn(jj)
          end if
        end do
      end do

      if (verbose) then
        write (6, *) 'normal end =', m
      end if
    end do

  end subroutine

  subroutine calc_lwa(qgpv1, qref1, imax, jmax, n, wx, wy, waa1, wac1, verbose)
    implicit none
    ! ### Model dimensions ###
    ! f2py integer, intent(hide):: imax, jmax, n
    integer, intent(in):: imax, jmax, n
    real, intent(in):: wx, wy   ! 28000 km x 28000 km
    real, intent(in):: qgpv1(imax, jmax, n), qref1(jmax, n)
    logical, optional  :: verbose
    real qe(imax, jmax)
    real, intent(out):: waa1(imax, jmax, n), wac1(imax, jmax, n)
    real :: dx, dy
    integer :: i,j,jjj,m

    if (.not. present(verbose)) verbose = .FALSE.

    dx = wx/real(imax)
    dy = wy/real(jmax - 1)

    ! ### Time loop ###
    do m = 1, n
        do jjj = 1, jmax
          ! ### Computing anticyclonic and cyclonic LWA at latitude jjj ###
          !         jjj = 64   ! at the center of channel
          ! add a loop here through jjj
          ! ### Compute qe ###
          qe(:, :) = qgpv1(:, :, m) - qref1(jjj, m)

          do i = 1, imax  !2
              waa1(i, jjj, m) = 0.
              wac1(i, jjj, m) = 0.
              do j = 1, jmax  !3
                if (j .lt. jjj .and. qe(i, j) .gt. 0.) then
                    wac1(i, jjj, m) = wac1(i, jjj, m) + qe(i, j)*dy
                end if
                if (j .ge. jjj .and. qe(i, j) .le. 0.) then
                    waa1(i, jjj, m) = waa1(i, jjj, m) - qe(i, j)*dy
                end if
              end do !3
          end do !2
        end do !jjj loop
        if (verbose) then
          write (6, *) 'normal end =', m
        end if
    end do

  end subroutine
end module
