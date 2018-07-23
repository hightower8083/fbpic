import numpy as np
from mako.template import Template
from fbpic.particles.templater import __path__

src_path = __path__[0]+'/'

E_arg = "                           Er_m{0}, Et_m{0}, Ez_m{0},"
B_arg = "                           Br_m{0}, Bt_m{0}, Bz_m{0},"

E_docstr = \
"""    Er_m{0}, Et_m{0}, Ez_m{0} : 2darrays of complexs
        The electric fields on the interpolation grid for the mode {0}"""

B_docstr = \
"""    Br_m{0}, Bt_m{0}, Bz_m{0} : 2darrays of complexs
        The magnetic fields on the interpolation grid for the mode {0}"""


mode_src = \
"""
        # E-Field
        # -------
        Fr = 0.
        Ft = 0.
        Fz = 0.
        # Add contribution from mode m
        Fr, Ft, Fz = add_linear_gather_for_mode( {0},
            Fr, Ft, Fz, exptheta_m, Er_m{0}, Et_m{0}, Ez_m{0},
            iz_lower, iz_upper, ir_lower, ir_upper,
            S_ll, S_lu, S_lg, S_ul, S_uu, S_ug )
        # Convert to Cartesian coordinates
        # and write to particle field arrays
        Ex += cos*Fr - sin*Ft
        Ey += sin*Fr + cos*Ft
        Ez += Fz

        # B-Field
        # -------
        # Clear the placeholders for the
        # gathered field for each coordinate
        Fr = 0.
        Ft = 0.
        Fz = 0.
        # Add contribution from mode m
        Fr, Ft, Fz = add_linear_gather_for_mode( {0},
            Fr, Ft, Fz, exptheta_m, Br_m{0}, Bt_m{0}, Bz_m{0},
            iz_lower, iz_upper, ir_lower, ir_upper,
            S_ll, S_lu, S_lg, S_ul, S_uu, S_ug )
        # Convert to Cartesian coordinates
        # and write to particle field arrays
        Bx += cos*Fr - sin*Ft
        By += sin*Fr + cos*Ft
        Bz += Fz

        # Increase index of the azimuthal complex factor
        exptheta_m *= exptheta_1
"""


def make_gather_push_cuda( Nm, module_file='./_gather_push_cuda_.py',
                 template_file='template_gather_push_cuda.py'):


    kernel_template = Template(filename=src_path+template_file)
    mrange = range(0, Nm)

    Args = {}
    Args['E_args'] = '\n'.join([E_arg.format(m) for m in mrange])
    Args['B_args'] = '\n'.join([B_arg.format(m) for m in mrange])

    Args['E_docstr'] = '\n\n'.join([E_docstr.format(m) for m in mrange])
    Args['B_docstr'] = '\n\n'.join([E_docstr.format(m) for m in mrange])

    Args['add_mode'] = ''.join([mode_src.format(m) for m in mrange])

    f = open(module_file, mode='w')
    f.writelines( kernel_template.render(**Args) )
    f.close()


