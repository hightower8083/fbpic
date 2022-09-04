# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the structure and methods associated with atomic ionization.
"""

import numpy as np
from scipy.constants import c, e, m_e, physical_constants
from .numba_methods import *
from ..cuda_numba_utils import allocate_empty, reallocate_and_copy_old, \
                                perform_cumsum_2d

# Check if CUDA is available, then import CUDA functions
from fbpic.utils.cuda import cuda_installed
from fbpic.utils.printing import catch_gpu_memory_error
if cuda_installed:
    import cupy
    from fbpic.utils.cuda import cuda_tpb_bpg_1d
    from .cuda_methods import *

class Radiator(object):
    """
    Class that contains the data associated with radiation
    and has method to calculate its spectrum, energy and angle.

    Main attributes
    ---------------
    - ionization_level: 1darray of integers (one element per particle)
      which contains the ionization state of each particle
    - w_times_level: 1darray of floats (one element per particle)
      which contains the number of physical particle that correspond to each
      macroparticle, multiplied by the ionization level. (This is updated
      whenever further ionization happens, and is passed to the deposition
      kernel as the effective weight of the particles)
    """
    def __init__(self, element, radiating_species, target_grid):
        """
        Initialize an Ionizer instance

        Parameters
        ----------
        radiating_species: an fbpic.Particles object
            This object is not modified or registered.
            It is only used in order to pass a number of additional argument.

        target_grid: a 3D array
        """
        # Register a few parameters
        self.use_cuda = radiating_species.use_cuda
        # Process ionized particles into batches
        self.batch_size = 10

        # Initialize radiation-relevant meta-data
        self.initialize_SR_parameters( target_grid, radiating_species.dt )

        # Initialize the required arrays
        Ntot = radiating_species.Ntot

    def initialize_SR_parameters( self, dt ):
        """
        Initialize parameters needed for the calculation of synchrotron
        radiation

        Parameters
        ----------
        dt: float (in seconds)
            The timestep of the simulation.
        """

        # Calculate the SR prefactors
        # - Scalars
        alpha = physical_constants['fine-structure constant'][0]
        r_e = physical_constants['classical electron radius'][0]
        wa = alpha**3 * c / r_e
        Ea = m_e*c**2/e * alpha**4/r_e
        # - Arrays (one element per ionization level)
        UH = get_ionization_energies('H')[0]
        Z = np.arange( len(Uion) ) + 1
        n_eff = Z * np.sqrt( UH/Uion )
        l_eff = n_eff[0] - 1
        C2 = 2**(2*n_eff) / (n_eff * gamma(n_eff+l_eff+1) * gamma(n_eff-l_eff))
        # For now, we assume l=0, m=0
        self.adk_power = - (2*n_eff - 1)
        self.adk_prefactor = dt * wa * C2 * ( Uion/(2*UH) ) \
            * ( 2*(Uion/UH)**(3./2)*Ea )**(2*n_eff - 1)
        self.adk_exp_prefactor = -2./3 * ( Uion/UH )**(3./2) * Ea


    @catch_gpu_memory_error
    def handle_ionization( self, ion ):
        """
        Handle ionization, either on CPU or GPU

        - For each ion macroparticle, decide whether it is going to
          be further ionized during this timestep, based on the ADK rate.
        - Add the electrons created from ionization to the `target_species`

        Parameters:
        -----------
        ion: an fbpic.Particles object
            The ionizable species, from which new electrons are created.
        """
        # Skip this function if there are no ions
        if ion.Ntot == 0:
            return
        
        # Process particles in batches (of typically 10, 20 particles)
        N_batch = int( ion.Ntot / self.batch_size ) + 1
        # Short-cuts
        use_cuda = self.use_cuda

        # Set the number of levels that should be distinguished
        if self.store_electrons_per_level:
            n_levels = self.level_max - self.level_start
        else:
            n_levels = 1

        # Create temporary arrays (on CPU or GPU, depending on `use_cuda`)
        ionized_from = allocate_empty( ion.Ntot, use_cuda, dtype=np.int16 )
        n_ionized = allocate_empty( (n_levels, N_batch), use_cuda,
                                    dtype=np.int64 )
        # Draw random numbers
        if self.use_cuda:
            random_draw = cupy.random.rand( ion.Ntot, dtype=cupy.float32 )
        else:
            random_draw = np.random.rand( ion.Ntot )

        # Determine the ions that are ionized, and count them in each batch
        # (one thread per batch on GPU; parallel loop over batches on CPU)
        if use_cuda:
            batch_grid_1d, batch_block_1d = cuda_tpb_bpg_1d( N_batch )
            ionize_ions_cuda[ batch_grid_1d, batch_block_1d ](
                N_batch, self.batch_size, ion.Ntot,
                self.level_start, self.level_max, n_levels,
                n_ionized, ionized_from, self.ionization_level, random_draw,
                self.adk_prefactor, self.adk_power, self.adk_exp_prefactor,
                ion.ux, ion.uy, ion.uz, ion.Ex, ion.Ey, ion.Ez,
                ion.Bx, ion.By, ion.Bz, ion.w, self.w_times_level )
        else:
            ionize_ions_numba(
                N_batch, self.batch_size, ion.Ntot,
                self.level_start, self.level_max, n_levels,
                n_ionized, ionized_from, self.ionization_level, random_draw,
                self.adk_prefactor, self.adk_power, self.adk_exp_prefactor,
                ion.ux, ion.uy, ion.uz, ion.Ex, ion.Ey, ion.Ez,
                ion.Bx, ion.By, ion.Bz, ion.w, self.w_times_level )

        # Count the total number of new electrons 
        cumulative_n_ionized = perform_cumsum_2d( n_ionized, use_cuda )
        # If no new particle was created, skip the rest of this function
        if use_cuda:
            if cupy.all( cumulative_n_ionized[:,-1] == 0 ):
                return
        else:
            if np.all( cumulative_n_ionized[:,-1] == 0 ):
                return

        # Loop over the electron species associated to each level
        # (when store_electrons_per_level is False, there is a single species)
        # Reallocate electron species (on CPU or GPU depending on `use_cuda`),
        # to accomodate the electrons produced by ionization,
        # and copy the old electrons to the new arrays
        assert len(self.target_species) == n_levels
        for i_level, elec in enumerate(self.target_species):
            old_Ntot = elec.Ntot
            # Cast to int transfers the data from the GPU if needed
            new_Ntot = old_Ntot + int( cumulative_n_ionized[i_level,-1] )
            reallocate_and_copy_old( elec, use_cuda, old_Ntot, new_Ntot )
            # Create the new electrons from ionization (one thread per batch)
            if use_cuda:
                copy_ionized_electrons_cuda[ batch_grid_1d, batch_block_1d ](
                    N_batch, self.batch_size, old_Ntot, ion.Ntot,
                    cumulative_n_ionized, ionized_from,
                    i_level, self.store_electrons_per_level,
                    elec.x, elec.y, elec.z, elec.inv_gamma,
                    elec.ux, elec.uy, elec.uz, elec.w,
                    elec.Ex, elec.Ey, elec.Ez, elec.Bx, elec.By, elec.Bz,
                    ion.x, ion.y, ion.z, ion.inv_gamma,
                    ion.ux, ion.uy, ion.uz, ion.w,
                    ion.Ex, ion.Ey, ion.Ez, ion.Bx, ion.By, ion.Bz )
                # Mark the new electrons as unsorted
                elec.sorted = False
            else:
                copy_ionized_electrons_numba(
                    N_batch, self.batch_size, old_Ntot, ion.Ntot,
                    cumulative_n_ionized, ionized_from,
                    i_level, self.store_electrons_per_level,
                    elec.x, elec.y, elec.z, elec.inv_gamma,
                    elec.ux, elec.uy, elec.uz, elec.w,
                    elec.Ex, elec.Ey, elec.Ez, elec.Bx, elec.By, elec.Bz,
                    ion.x, ion.y, ion.z, ion.inv_gamma,
                    ion.ux, ion.uy, ion.uz, ion.w,
                    ion.Ex, ion.Ey, ion.Ez, ion.Bx, ion.By, ion.Bz )

            # If the electrons are tracked, generate new ids
            # (on GPU or GPU depending on `use_cuda`)
            generate_new_ids( elec, old_Ntot, new_Ntot )


    def send_to_gpu( self ):
        """
        Copy the ionization data to the GPU.
        """
        if self.use_cuda:
            # Arrays with one element per macroparticles
            self.ionization_level = cupy.asarray( self.ionization_level )
            self.w_times_level = cupy.asarray( self.w_times_level )
            # Small-size arrays with ADK parameters
            # (One element per ionization level)
            self.adk_power = cupy.asarray( self.adk_power )
            self.adk_prefactor = cupy.asarray( self.adk_prefactor )
            self.adk_exp_prefactor = cupy.asarray( self.adk_exp_prefactor )

    def receive_from_gpu( self ):
        """
        Receive the ionization data from the GPU.
        """
        if self.use_cuda:
            # Arrays with one element per macroparticles
            self.ionization_level = self.ionization_level.get()
            self.w_times_level = self.w_times_level.get()
            # Small-size arrays with ADK parameters
            # (One element per ionization level)
            self.adk_power = self.adk_power.get()
            self.adk_prefactor = self.adk_prefactor.get()
            self.adk_exp_prefactor = self.adk_exp_prefactor.get()
