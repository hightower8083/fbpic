# Copyright 2019, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)

It defines the picmi Simulation interface
"""
import numpy as np
from scipy.constants import c, e, m_e
from .particle_charge_and_mass import particle_charge, particle_mass

# Import relevant fbpic object
from fbpic.main import Simulation as FBPICSimulation
from fbpic.fields.smoothing import BinomialSmoother
from fbpic.lpa_utils.laser import add_laser_pulse, GaussianLaser
from fbpic.lpa_utils.bunch import add_elec_bunch_gaussian
from fbpic.openpmd_diag import FieldDiagnostic, ParticleDiagnostic

# Import picmi base class
from picmistandard import PICMI_Simulation, PICMI_CylindricalGrid
from picmistandard import PICMI_AnalyticDistribution, PICMI_GriddedLayout
from picmistandard import PICMI_PseudoRandomLayout, PICMI_GaussianBunchDistribution
from picmistandard import PICMI_LaserAntenna, PICMI_GaussianLaser
from picmistandard import PICMI_Species, PICMI_MultiSpecies
from picmistandard import PICMI_FieldDiagnostic, PICMI_ParticleDiagnostic

# Define a new simulation object for picmi, that derives from PICMI_Simulation
class Simulation( PICMI_Simulation ):

    # Redefine the `init` method, as required by the picmi `_ClassWithInit`
    def init(self, kw):

        # Get the grid
        grid = self.solver.grid
        if not type(grid) == PICMI_CylindricalGrid:
            raise ValueError('When using fbpic with PICMI, '
                'the grid needs to be a CylindricalGrid object.')
        # Check rmin and boundary conditions
        assert grid.rmin == 0.
        assert grid.bc_zmin == grid.bc_zmax
        assert grid.bc_zmax in ['periodic', 'open']
        assert grid.bc_rmax == 'reflective'

        # Determine timestep
        if self.solver.cfl is not None:
            dz = (grid.zmax-grid.zmin)/grid.nz
            dt = self.solver.cfl * dz / c
        elif self.time_step_size is not None:
            dt = self.time_step_size
        else:
            raise ValueError(
                'You need to either set the `cfl` of the solver\n'
                'or the `timestep_size` of the `Simulation`.')

        # Convert API for the smoother
        if self.solver.source_smoother is None:
            smoother = BinomialSmoother()
        else:
            smoother = BinomialSmoother(
                n_passes=self.solver.source_smoother.n_pass,
                compensator=self.solver.source_smoother.compensation )

        # Initialize and store the FBPIC simulation object
        self.fbpic_sim = FBPICSimulation(
            Nz=int(grid.nz), zmin=grid.zmin, zmax=grid.zmax,
            Nr=int(grid.nr), rmax=grid.rmax, Nm=grid.n_azimuthal_modes,
            dt=dt, use_cuda=True, boundaries=grid.bc_zmax,
            smoother=smoother )
        # Remove default electron species
        self.fbpic_sim.ptcl = []
        # Set the moving window
        if grid.moving_window_velocity is not None:
            self.fbpic_sim.set_moving_window(grid.moving_window_velocity[-1])


    # Redefine the method `add_laser` from the PICMI Simulation class
    def add_laser( self, laser, injection_method ):
        # Call method of parent class
        PICMI_Simulation.add_laser( self, laser, injection_method )

        # Handle injection method
        assert type(injection_method) == PICMI_LaserAntenna
        # Handle laser profile method
        if type(laser) == PICMI_GaussianLaser:
            assert laser.propagation_direction[0] == 0.
            assert laser.propagation_direction[1] == 0.
            assert (laser.zeta is None) or (laser.zeta == 0)
            assert (laser.beta is None) or (laser.beta == 0)
            phi2_chirp = laser.phi2
            if phi2_chirp is None:
                phi2_chirp = 0
            laser_profile = GaussianLaser( a0=laser.a0, waist=laser.waist,
                z0=laser.centroid_position[-1], zf=laser.focal_position[-1],
                tau=laser.duration, theta_pol=laser.polarization_angle,
                phi2_chirp=phi2_chirp )
        else:
            raise ValueError('Unknown laser profile: %s' %type(injection_method))

        # Inject the laser
        add_laser_pulse( self.fbpic_sim, laser_profile, method='antenna',
            z0_antenna=injection_method.position[-1] )


    # Redefine the method `add_species` from the PICMI Simulation class
    def add_species( self, species, layout, initialize_self_field=False ):
        # Call method of parent class
        PICMI_Simulation.add_species( self, species, layout,
                                      initialize_self_field )

        # Extract list of species
        if type(species) == PICMI_Species:
            species_instances_list = [species]
        elif type(species) == PICMI_MultiSpecies:
            species_instances_list = species.species_instances_list
        else:
            raise ValueError('Unknown type: %s' %type(species))

        # Loop over species and create FBPIC species
        for s in species_instances_list:

            # Get their charge and mass
            if s.particle_type is not None:
                s.charge = particle_charge[s.particle_type]
                s.mass = particle_mass[s.particle_type]
            # If `charge_state` is set, redefine the charge and mass
            if s.charge_state is not None:
                s.charge = s.charge_state*e
                s.mass -= s.charge_state*m_e

            # Add the species to the FBPIC simulation
            fbpic_species = self._create_new_fbpic_species(s,
                                        layout, initialize_self_field)

            # Register a pointer to the FBPIC species in the PICMI species itself
            # (Useful for particle diagnostics later on)
            s.fbpic_species = fbpic_species

        # Loop over species and handle ionization
        for s in species_instances_list:
            for interaction in s.interactions:
                assert interaction[0] == 'ionization'
                assert interaction[1] == 'ADK'
                picmi_target = interaction[2]
                if not hasattr( picmi_target, 'fbpic_species' ):
                    raise RunTimeError('For ionization with PICMI+FBPIC:\n'
                        'You need to add the target species to the simulation,'
                        ' before the other species.')
                fbpic_target = picmi_target.fbpic_species
                fbpic_source = s.fbpic_species
                fbpic_source.make_ionizable( element=s.particle_type,
                                             level_start=s.charge_state,
                                             target_species=fbpic_target )


    def _create_new_fbpic_species(self, s, layout, initialize_self_field):

        # - For the case of a plasma defined in a gridded layout
        if (type(s.initial_distribution)==PICMI_AnalyticDistribution) and \
            (type(layout) == PICMI_GriddedLayout):
            import numexpr
            def dens_func(z, r):
                n = numexpr.evaluate(s.initial_distribution.density_expression)
                return n
            p_nr = layout.n_macroparticle_per_cell[0]
            p_nt = layout.n_macroparticle_per_cell[1]
            p_nz = layout.n_macroparticle_per_cell[2]
            fbpic_species = self.fbpic_sim.add_new_species(
                q=s.charge, m=s.mass, n=1.,
                dens_func=dens_func, p_nz=p_nz, p_nr=p_nr, p_nt=p_nt,
                continuous_injection=s.initial_distribution.fill_in )

        # - For the case of a Gaussian beam
        elif (type(s.initial_distribution)==PICMI_GaussianBunchDistribution) and \
            (type(layout) == PICMI_PseudoRandomLayout):
            assert s.charge == -e and s.mass == m_e
            assert initialize_self_field
            dist = s.initial_distribution
            Q = dist.n_physical_particles * e
            sig_r = dist.rms_bunch_size[0]
            sig_z = dist.rms_bunch_size[-1]
            gamma0_beta0 = dist.centroid_velocity[-1]
            gamma0 = np.sqrt( 1 + gamma0_beta0**2 )
            n_emit = sig_r * dist.rms_velocity[0]
            sig_gamma = dist.rms_velocity[-1]/c
            zf = dist.centroid_position[-1]
            add_elec_bunch_gaussian( self.fbpic_sim,
                gamma0=gamma0, sig_gamma=sig_gamma,
                sig_r=sig_r, sig_z=sig_z, n_emit=n_emit, Q=Q,
                N=layout.n_macroparticles, zf=zf )
            fbpic_species = self.fbpic_sim.ptcl[-1]

        else:
            raise ValueError('Unknown combination of layout and distribution')

        return fbpic_species


    # Redefine the method `add_diagnostic` of the parent class
    def add_diagnostic(self, diagnostic):
        # Call method of parent class
        PICMI_Simulation.add_diagnostic( self, diagnostic )

        # Handle diagnostic
        if diagnostic.step_min is None:
            iteration_min = 0
        else:
            iteration_min = diagnostic.step_min
        if diagnostic.step_max is None:
            iteration_max = np.inf
        else:
            iteration_max = diagnostic.step_max
        # Register field diagnostic
        if type(diagnostic) == PICMI_FieldDiagnostic:
            diag = FieldDiagnostic(
                    period=diagnostic.period,
                    fldobject=self.fbpic_sim.fld,
                    comm=self.fbpic_sim.comm,
                    fieldtypes=diagnostic.data_list,
                    write_dir=diagnostic.write_dir,
                    iteration_min=iteration_min,
                    iteration_max=iteration_max)
        # Register particle diagnostic
        elif type(diagnostic) == PICMI_ParticleDiagnostic:
            species_dict = {}
            for s in diagnostic.species:
                if s.name is None:
                    raise ValueError('When using a species in a diagnostic, '
                                      'its name must be set.')
                species_dict[s.name] = s.fbpic_species
            diag = ParticleDiagnostic(
                    period=diagnostic.period,
                    species=species_dict,
                    comm=self.fbpic_sim.comm,
                    particle_data=diagnostic.data_list,
                    write_dir=diagnostic.write_dir,
                    iteration_min=iteration_min,
                    iteration_max=iteration_max)

        # Add it to the FBPIC simulation
        self.fbpic_sim.diags.append( diag )


    # Redefine the method `step` of the parent class
    def step(self, nsteps):
        self.fbpic_sim.step( nsteps )
