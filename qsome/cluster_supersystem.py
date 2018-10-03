# A method to define a cluster supersystem
# Daniel Graham

import os
from qsome import supersystem
from pyscf import gto

class ClusterSuperSystem(supersystem.SuperSystem):

    def __init__(self, subsystems, ct_method, proj_oper='huz', filename=None,
                 ft_cycles=100, ft_conv=1e-8, ft_grad=1e-8, ft_diis=1, 
                 ft_setfermi=0, ft_initguess='minao', ft_updatefock=0, 
                 cycles=100, conv=1e-8, grad=1e-8, damp=0, shift=0, 
                 smearsigma=0, initguess='minao', includeghost=False, 
                 grid=4, verbose=3, analysis=False, debug=False):

        self.subsystems = subsystems
        self.ct_method = ct_method
        self.proj_oper = proj_oper

        if filename == None:
            filename = os.getcwd() + '/temp.inp'
        self.filename = filename

        # freeze and thaw settings
        self.ft_cycles = ft_cycles
        self.ft_conv = ft_conv
        self.ft_grad = ft_grad
        self.ft_diis = ft_diis
        self.ft_setfermi = ft_setfermi
        self.ft_initguess = ft_initguess
        self.ft_updatefock = ft_updatefock

        # charge transfer settings
        self.cycles = cycles
        self.conv = conv
        self.grad = grad
        self.damp = damp
        self.shift = shift
        self.smearsigma = smearsigma
        self.initguess = initguess
        self.includeghost = includeghost

        # general system settings
        self.grid = grid
        self.verbose = verbose
        self.analysis = analysis
        self.debug = debug

        self.concat_mols()
        self.init_density()

    def sub2sup(self):
        pass

    def init_density(self):
        pass

    def concat_mols(self):

        # this works but can be done MUCH better.
        self.mol = gto.Mole()
        self.mol.basis = {}
        atm = []
        nghost = 0
        for i in range(len(self.subsystems)):
            subsystem = self.subsystems[i]
            if i <= 0:
                self.mol.charge = subsystem.mol.charge
                self.mol.spin = subsystem.mol.spin
                # copy all atoms from mole A
                for j in range(subsystem.mol.natm):
                    if 'ghost' in subsystem.mol.atom_symbol(j).lower():
                        if self.includeghost:
                            nghost += 1
                            ghost_name = subsystem.mol.atom_symbol(j).split(':')[0] + f':{nghost}'
                            atm.append([ghost_name, subsystem.mol.atom_coord(j)])
                            self.mol.basis.update({ghost_name: subsystem.mol.basis[subsystem.mol.atom_symbol(j)]})
                    else:
                        atm.append([subsystem.mol.atom_symbol(j), subsystem.mol.atom_coord(j)])
                        if isinstance(subsystem.mol.basis, str): # If subsystems are created by user, rather than read from input.
                            self.mol.basis.update({subsystem.mol.atom_symbol(j): gto.basis.load(subsystem.mol.basis, subsystem.mol.atom_symbol(j))})
                        else:
                            self.mol.basis.update({subsystem.mol.atom_symbol(j): subsystem.mol.basis[subsystem.mol.atom_symbol(j)]})

            # copy all atoms from other subsystems
            else:
                self.mol.charge += subsystem.mol.charge
                self.mol.spin += subsystem.mol.spin
                for j in range(subsystem.mol.natm):
                    if 'ghost' in subsystem.mol.atom_symbol(j).lower():
                        if self.includeghost:
                            nghost += 1
                            oghost = int(subsystem.mol.atom_symbol(j).split(':')[1])
                            newsym = f'GHOST:{nghost}'
                            atm.append([newsym, subsystem.mol.atom_coord(j)])
                            self.mol.basis.update({newsym.lower(): subsystem.mol.basis[f'ghost:{oghost}']})
                    else:
                        #Difference here. Maybe not 1
                        atm.append([subsystem.mol.atom_symbol(j)+':1', subsystem.mol.atom_coord(j)])
                        if isinstance(subsystem.mol.basis, str): #Manually loaded basis.
                            self.mol.basis.update({subsystem.mol.atom_symbol(j)+':1': gto.basis.load(subsystem.mol.basis, subsystem.mol.atom_symbol(j))})
                        else:
                            self.mol.basis.update({subsystem.mol.atom_symbol(j)+':1': subsystem.mol.basis[subsystem.mol.atom_symbol(j)]})

        self.mol.atom = atm
        self.mol.verbose = self.verbose
        self.mol.unit = 'bohr' # atom_coord is always stored in bohr for some reason. Trust me this is right.
        self.mol.build(dump_input=False)

    def supermolecular_energy(self):
        pass

    def env_in_env_energy(self):
        pass

    def get_proj_op(self):
        pass

    def get_embedding_pot(self):
        pass

    def update_fock(self):
        pass

    def save_chkfile(self):
        pass

    def freeze_and_thaw(self):
        pass
