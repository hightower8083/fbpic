# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters, Igor Andriyash
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the methods specific for gathering, pushing and copying the 
lightweight species on the GPU using CUDA.
"""
from numba import cuda
from scipy.constants import c, e
import math
# Import inline functions
from ..gathering.inline_functions import \
    add_linear_gather_for_mode, add_cubic_gather_for_mode
from ..push.inline_functions import push_p_vay
from .inline_functions import copy_ionized_electrons_batch_lightweight


# Compile the inline functions for GPU
add_linear_gather_for_mode = cuda.jit( add_linear_gather_for_mode,
                                        device=True, inline=True )
add_cubic_gather_for_mode = cuda.jit( add_cubic_gather_for_mode,
                                        device=True, inline=True )
push_p_vay = cuda.jit( push_p_vay, device=True, inline=True )
copy_ionized_electrons_batch_lightweight = cuda.jit( \
         copy_ionized_electrons_batch_lightweight, device=True, inline=True )

# --------
# Copying
# --------
@cuda.jit
def copy_ionized_electrons_cuda_lightweight(
    N_batch, batch_size, elec_old_Ntot, ion_Ntot,
    cumulative_n_ionized, ionized_from,
    i_level, store_electrons_per_level,
    elec_x, elec_y, elec_z, elec_inv_gamma,
    elec_ux, elec_uy, elec_uz, elec_w,
    ion_x, ion_y, ion_z, ion_inv_gamma,
    ion_ux, ion_uy, ion_uz, ion_w ):
    """
    Create the new lightweight electrons by copying the properties (position,
    momentum, etc) of the ions that they originate from.
    """
    # Select the current batch
    i_batch = cuda.grid(1)
    if i_batch < N_batch:
        copy_ionized_electrons_batch_lightweight(
            i_batch, batch_size, elec_old_Ntot, ion_Ntot,
            cumulative_n_ionized, ionized_from,
            i_level, store_electrons_per_level,
            elec_x, elec_y, elec_z, elec_inv_gamma,
            elec_ux, elec_uy, elec_uz, elec_w,
            ion_x, ion_y, ion_z, ion_inv_gamma,
            ion_ux, ion_uy, ion_uz, ion_w )

# -----------------------
# Field gathering linear
# -----------------------
@cuda.jit
def gather_push_gpu_linear(x, y, z, ux, uy, uz,
                           inv_gamma, q, m, Ntot, dt_p,
                           dt_x, x_push, y_push, z_push,
                           invdz, zmin, Nz,
                           invdr, rmin, Nr,
                           Er_m0, Et_m0, Ez_m0,
                           Er_m1, Et_m1, Ez_m1,
                           Br_m0, Bt_m0, Bz_m0,
                           Br_m1, Bt_m1, Bz_m1 ):
    """
    - Gathering of the fields (E and B) using numba on the GPU.
      Iterates over the particles, calculates the weighted amount
      of fields acting on each particle based on its shape (linear).
      Fields are gathered in cylindrical coordinates and then
      transformed to cartesian coordinates.
      Supports only mode 0 and 1.
    - Advance the particles' momenta over `dt_p`
    - Advance the particles' positions over `dt_x` using the momenta
      ux, uy, uz, multiplied by the scalar coefficients x_push, y_push, z_push.

    Parameters
    ----------
    x, y, z : 1darray of floats (in meters)
        The position of the particles

    ux, uy, uz : 1darray of floats
        The velocity of the particles
        (is modified by this function)

    inv_gamma : 1darray of floats
        The inverse of the relativistic gamma factor

    q : float
        The charge of the particle species

    m : float
        The mass of the particle species

    Ntot : int
        The total number of particles

    dt_p : float
        The time by which the momenta is advanced

    dt_x : float (seconds)
        The timestep by which the position is advanced

    x_push, y_push, z_push: float, dimensionless
        Multiplying coefficient for the momenta in x, y and z
        e.g. if x_push=1., the particles are pushed forward in x
             if x_push=-1., the particles are pushed backward in x

    invdz, invdr : float (in meters^-1)
        Inverse of the grid step along the considered direction

    zmin, rmin : float (in meters)
        Position of the edge of the simulation box along the
        direction considered

    Nz, Nr : int
        Number of gridpoints along the considered direction

    Er_m0, Et_m0, Ez_m0 : 2darray of complexs
        The electric fields on the interpolation grid for the mode 0

    Er_m1, Et_m1, Ez_m1 : 2darray of complexs
        The electric fields on the interpolation grid for the mode 1

    Br_m0, Bt_m0, Bz_m0 : 2darray of complexs
        The magnetic fields on the interpolation grid for the mode 0

    Br_m1, Bt_m1, Bz_m1 : 2darray of complexs
        The magnetic fields on the interpolation grid for the mode 1

    """
    # Set few constants
    econst = q*dt_p/(m*c)
    bconst = 0.5*q*dt_p/m
    cdt = c*dt_x
    # Get the 1D CUDA grid
    i = cuda.grid(1)
    # Deposit the field per cell in parallel
    # (for threads < number of particles)
    if i < x.shape[0]:
        # Preliminary arrays for the cylindrical conversion
        # --------------------------------------------
        # Position
        xj = x[i]
        yj = y[i]
        zj = z[i]

        # Cylindrical conversion
        rj = math.sqrt( xj**2 + yj**2 )
        if (rj !=0. ) :
            invr = 1./rj
            cos = xj*invr  # Cosine
            sin = yj*invr  # Sine
        else :
            cos = 1.
            sin = 0.
        exptheta_m0 = 1.
        exptheta_m1 = cos - 1.j*sin

        # Get linear weights for the deposition
        # --------------------------------------------
        # Positions of the particles, in the cell unit
        r_cell =  invdr*(rj - rmin) - 0.5
        z_cell =  invdz*(zj - zmin) - 0.5
        # Original index of the uppper and lower cell
        ir_lower = int(math.floor( r_cell ))
        ir_upper = ir_lower + 1
        iz_lower = int(math.floor( z_cell ))
        iz_upper = iz_lower + 1
        # Linear weight
        Sr_lower = ir_upper - r_cell
        Sr_upper = r_cell - ir_lower
        Sz_lower = iz_upper - z_cell
        Sz_upper = z_cell - iz_lower
        # Set guard weights to zero
        Sr_guard = 0.

        # Treat the boundary conditions
        # --------------------------------------------
        # guard cells in lower r
        if ir_lower < 0:
            Sr_guard = Sr_lower
            Sr_lower = 0.
            ir_lower = 0
        # absorbing in upper r
        if ir_lower > Nr-1:
            ir_lower = Nr-1
        if ir_upper > Nr-1:
            ir_upper = Nr-1
        # periodic boundaries in z
        # lower z boundaries
        if iz_lower < 0:
            iz_lower += Nz
        if iz_upper < 0:
            iz_upper += Nz
        # upper z boundaries
        if iz_lower > Nz-1:
            iz_lower -= Nz
        if iz_upper > Nz-1:
            iz_upper -= Nz

        # Precalculate Shapes
        S_ll = Sz_lower*Sr_lower
        S_lu = Sz_lower*Sr_upper
        S_ul = Sz_upper*Sr_lower
        S_uu = Sz_upper*Sr_upper
        S_lg = Sz_lower*Sr_guard
        S_ug = Sz_upper*Sr_guard

        # E-Field
        # -------
        Fr = 0.
        Ft = 0.
        Fz = 0.
        # Add contribution from mode 0
        Fr, Ft, Fz = add_linear_gather_for_mode( 0,
            Fr, Ft, Fz, exptheta_m0, Er_m0, Et_m0, Ez_m0,
            iz_lower, iz_upper, ir_lower, ir_upper,
            S_ll, S_lu, S_lg, S_ul, S_uu, S_ug )
        # Add contribution from mode 1
        Fr, Ft, Fz = add_linear_gather_for_mode( 1,
            Fr, Ft, Fz, exptheta_m1, Er_m1, Et_m1, Ez_m1,
            iz_lower, iz_upper, ir_lower, ir_upper,
            S_ll, S_lu, S_lg, S_ul, S_uu, S_ug )
        # Convert to Cartesian coordinates
        # and write to particle field arrays
        Ex = cos*Fr - sin*Ft
        Ey = sin*Fr + cos*Ft
        Ez = Fz

        # B-Field
        # -------
        # Clear the placeholders for the
        # gathered field for each coordinate
        Fr = 0.
        Ft = 0.
        Fz = 0.
        # Add contribution from mode 0
        Fr, Ft, Fz = add_linear_gather_for_mode( 0,
            Fr, Ft, Fz, exptheta_m0, Br_m0, Bt_m0, Bz_m0,
            iz_lower, iz_upper, ir_lower, ir_upper,
            S_ll, S_lu, S_lg, S_ul, S_uu, S_ug )
        # Add contribution from mode 1
        Fr, Ft, Fz = add_linear_gather_for_mode( 1,
            Fr, Ft, Fz, exptheta_m1, Br_m1, Bt_m1, Bz_m1,
            iz_lower, iz_upper, ir_lower, ir_upper,
            S_ll, S_lu, S_lg, S_ul, S_uu, S_ug )
        # Convert to Cartesian coordinates
        # and write to particle field arrays
        Bx = cos*Fr - sin*Ft
        By = sin*Fr + cos*Ft
        Bz = Fz

        # Pushing particle's momenta
        ux[i], uy[i], uz[i], inv_gamma[i] = push_p_vay(ux[i], uy[i], uz[i],
            inv_gamma[i], Ex, Ey, Ez, Bx, By, Bz, econst, bconst)

        # Pushing particle's position
        inv_g = inv_gamma[i]
        x[i] += cdt*x_push*inv_g*ux[i]
        y[i] += cdt*y_push*inv_g*uy[i]
        z[i] += cdt*z_push*inv_g*uz[i]

# -----------------------
# Field gathering cubic
# -----------------------

@cuda.jit
def gather_push_gpu_cubic(x, y, z, ux, uy, uz,
                           inv_gamma, q, m, Ntot, dt_p,
                           dt_x, x_push, y_push, z_push,
                           invdz, zmin, Nz,
                           invdr, rmin, Nr,
                           Er_m0, Et_m0, Ez_m0,
                           Er_m1, Et_m1, Ez_m1,
                           Br_m0, Bt_m0, Bz_m0,
                           Br_m1, Bt_m1, Bz_m1 ):
    """
    - Gathering of the fields (E and B) using numba on the GPU.
      Iterates over the particles, calculates the weighted amount
      of fields acting on each particle based on its shape (cubic).
      Fields are gathered in cylindrical coordinates and then
      transformed to cartesian coordinates.
      Supports only mode 0 and 1.
    - Advance the particles' momenta over `dt_p`
    - Advance the particles' positions over `dt_x` using the momenta
      ux, uy, uz, multiplied by the scalar coefficients x_push, y_push, z_push.

    Parameters
    ----------
    x, y, z : 1darray of floats (in meters)
        The position of the particles

    ux, uy, uz : 1darray of floats
        The velocity of the particles
        (is modified by this function)

    inv_gamma : 1darray of floats
        The inverse of the relativistic gamma factor

    q : float
        The charge of the particle species

    m : float
        The mass of the particle species

    Ntot : int
        The total number of particles

    dt_p : float
        The time by which the momenta is advanced

    dt_x : float (seconds)
        The timestep by which the position is advanced

    x_push, y_push, z_push: float, dimensionless
        Multiplying coefficient for the momenta in x, y and z
        e.g. if x_push=1., the particles are pushed forward in x
             if x_push=-1., the particles are pushed backward in x

    invdz, invdr : float (in meters^-1)
        Inverse of the grid step along the considered direction

    zmin, rmin : float (in meters)
        Position of the edge of the simulation box along the
        direction considered

    Nz, Nr : int
        Number of gridpoints along the considered direction

    Er_m0, Et_m0, Ez_m0 : 2darray of complexs
        The electric fields on the interpolation grid for the mode 0

    Er_m1, Et_m1, Ez_m1 : 2darray of complexs
        The electric fields on the interpolation grid for the mode 1

    Br_m0, Bt_m0, Bz_m0 : 2darray of complexs
        The magnetic fields on the interpolation grid for the mode 0

    Br_m1, Bt_m1, Bz_m1 : 2darray of complexs
        The magnetic fields on the interpolation grid for the mode 1

    """
    # Set few constants
    econst = q*dt_p/(m*c)
    bconst = 0.5*q*dt_p/m
    cdt = c*dt_x
    # Get the 1D CUDA grid
    i = cuda.grid(1)
    # Deposit the field per cell in parallel
    # (for threads < number of particles)
    if i < x.shape[0]:
        # Preliminary arrays for the cylindrical conversion
        # --------------------------------------------
        # Position
        xj = x[i]
        yj = y[i]
        zj = z[i]

        # Cylindrical conversion
        rj = math.sqrt(xj**2 + yj**2)
        if (rj != 0.):
            invr = 1./rj
            cos = xj*invr  # Cosine
            sin = yj*invr  # Sine
        else:
            cos = 1.
            sin = 0.
        exptheta_m0 = 1.
        exptheta_m1 = cos - 1.j*sin

        # Get weights for the deposition
        # --------------------------------------------
        # Positions of the particle, in the cell unit
        r_cell = invdr*(rj - rmin) - 0.5
        z_cell = invdz*(zj - zmin) - 0.5

        # Calculate the shape factors
        Sr = cuda.local.array((4,), dtype=float64)
        ir_lowest = int64(math.floor(r_cell)) - 1
        r_local = r_cell-ir_lowest
        Sr[0] = -1./6. * (r_local-2.)**3
        Sr[1] = 1./6. * (3.*(r_local-1.)**3 - 6.*(r_local-1.)**2 + 4.)
        Sr[2] = 1./6. * (3.*(2.-r_local)**3 - 6.*(2.-r_local)**2 + 4.)
        Sr[3] = -1./6. * (1.-r_local)**3
        Sz = cuda.local.array((4,), dtype=float64)
        iz_lowest = int64(math.floor(z_cell)) - 1
        z_local = z_cell-iz_lowest
        Sz[0] = -1./6. * (z_local-2.)**3
        Sz[1] = 1./6. * (3.*(z_local-1.)**3 - 6.*(z_local-1.)**2 + 4.)
        Sz[2] = 1./6. * (3.*(2.-z_local)**3 - 6.*(2.-z_local)**2 + 4.)
        Sz[3] = -1./6. * (1.-z_local)**3

        # E-Field
        # -------
        Fr = 0.
        Ft = 0.
        Fz = 0.
        # Add contribution from mode 0
        Fr, Ft, Fz = add_cubic_gather_for_mode( 0,
            Fr, Ft, Fz, exptheta_m0, Er_m0, Et_m0, Ez_m0,
            ir_lowest, iz_lowest, Sr, Sz, Nr, Nz )
        # Add contribution from mode 1
        Fr, Ft, Fz = add_cubic_gather_for_mode( 1,
            Fr, Ft, Fz, exptheta_m1, Er_m1, Et_m1, Ez_m1,
            ir_lowest, iz_lowest, Sr, Sz, Nr, Nz )
        # Convert to Cartesian coordinates
        # and write to particle field arrays
        Ex = cos*Fr - sin*Ft
        Ey = sin*Fr + cos*Ft
        Ez = Fz

        # B-Field
        # -------
        # Clear the placeholders for the
        # gathered field for each coordinate
        Fr = 0.
        Ft = 0.
        Fz = 0.
        # Add contribution from mode 0
        Fr, Ft, Fz =  add_cubic_gather_for_mode( 0,
            Fr, Ft, Fz, exptheta_m0, Br_m0, Bt_m0, Bz_m0,
            ir_lowest, iz_lowest, Sr, Sz, Nr, Nz )
        # Add contribution from mode 1
        Fr, Ft, Fz =  add_cubic_gather_for_mode( 1,
            Fr, Ft, Fz, exptheta_m1, Br_m1, Bt_m1, Bz_m1,
            ir_lowest, iz_lowest, Sr, Sz, Nr, Nz )
        # Convert to Cartesian coordinates
        # and write to particle field arrays
        Bx = cos*Fr - sin*Ft
        By = sin*Fr + cos*Ft
        Bz = Fz

        # Pushing particle's momenta
        ux[i], uy[i], uz[i], inv_gamma[i] = push_p_vay(ux[i], uy[i], uz[i],
            inv_gamma[i], Ex, Ey, Ez, Bx, By, Bz, econst, bconst)

        # Pushing particle's position
        inv_g = inv_gamma[i]
        x[i] += cdt * x_push*inv_g * ux[i]
        y[i] += cdt * y_push*inv_g * uy[i]
        z[i] += cdt * z_push*inv_g * uz[i]