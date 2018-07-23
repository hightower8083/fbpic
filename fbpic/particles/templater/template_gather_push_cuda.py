# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters, Igor Andriyash
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the field gathering methods linear and cubic order shapes
on the GPU using CUDA, for one azimuthal mode at a time
"""
from numba import cuda, float64, int64
from numba import float64, int64
from scipy.constants import c, e
import math
# Import inline functions
from fbpic.particles.gathering.inline_functions import add_linear_gather_for_mode
from fbpic.particles.gathering.inline_functions import add_cubic_gather_for_mode
from fbpic.particles.push.inline_functions import push_p_vay
# Compile the inline functions for GPU
add_linear_gather_for_mode = cuda.jit( add_linear_gather_for_mode,
                                        device=True, inline=True )
add_cubic_gather_for_mode = cuda.jit( add_cubic_gather_for_mode,
                                        device=True, inline=True )
push_p_vay = cuda.jit( push_p_vay, device=True, inline=True )


# -----------------------
# Field gathering linear
# -----------------------

@cuda.jit
def gather_field_gpu_linear_one_mode(x, y, z, ux, uy, uz,
                           inv_gamma, q, m, Ntot, dt_p,
                           dt_x, x_push, y_push, z_push,
                           invdz, zmin, Nz,
                           invdr, rmin, Nr, Nm,
${E_args}
${B_args} ):
    """
    - Gathering of the fields (E and B) using numba on the GPU.
      Iterates over the particles, calculates the weighted amount
      of fields acting on each particle based on its shape (linear).
      Fields are gathered in cylindrical coordinates and then
      transformed to cartesian coordinates.
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

${E_docstr}

${B_docstr}
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

        # Define field variables
        Ex = 0.
        Ey = 0.
        Ez = 0.
        Bx = 0.
        By = 0.
        Bz = 0.

        exptheta_m1 = (cos - 1.j*sin)
        exptheta_m = 1.
${add_mode}

        # Pushing particle's momenta
        ux[i], uy[i], uz[i], inv_gamma[i] = push_p_vay(ux[i], uy[i], uz[i],
            inv_gamma[i], Ex, Ey, Ez, Bx, By, Bz, econst, bconst)

        # Pushing particle's position
        inv_g = inv_gamma[i]
        x[i] += cdt * x_push*inv_g * ux[i]
        y[i] += cdt * y_push*inv_g * uy[i]
        z[i] += cdt * z_push*inv_g * uz[i]
