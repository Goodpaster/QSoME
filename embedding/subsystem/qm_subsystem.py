#Defines a quantum mechanical subsystem object.

from pyscf import scf

def save_chkfile(subsys, filename=None):
    if filename is None:
        if subsys.filename is None:
            return False
        filename = os.path.splitext(subsys.filename)[0] + '.hdf5'
    if os.path.isfile(filename):
        try:
            with h5py.File(filename, 'r+') as h5_file:
                subsys_coeff = h5_file['subsystem/mo_coeff']
                subsys_coeff[...] = subsys.mf_obj.mo_coeff
                subsys_occ = h5_file['subsystem/mo_occ']
                subsys_occ[...] = subsys.mo_occ
                subsys_energy = h5_file['subsystem/mo_energy']
                subsys_energy[...] = subsys.mo_energy
        except TypeError:
            logger.warn(subsys.mol, "Overwriting existing chkfile")
            with h5py.File(filename, 'w') as h5_file:
                sub_mol = h5_file.create_group('subsystem')
                sub_mol.create_dataset('mo_coeff', data=subsys.mo_coeff)
                sub_mol.create_dataset('mo_occ', data=subsys.mo_occ)
                sub_mol.create_dataset('mo_energy', data=subsys.mo_energy)
        except KeyError:
            logger.warn(subsys.mol, "Updating existing chkfile")
            with h5py.File(filename, 'a') as h5_file:
                sub_mol = h5_file.create_group(f'subsystem')
                sub_mol.create_dataset('mo_coeff', data=subsys.mo_coeff)
                sub_mol.create_dataset('mo_occ', data=subsys.mo_occ)
                sub_mol.create_dataset('mo_energy', data=subsys.mo_energy)

    else:
        with h5py.File(filename, 'w') as h5_file:
            sub_mol = h5_file.create_group(f'subsystem')
            sub_mol.create_dataset('mo_coeff', data=subsys.mo_coeff)
            sub_mol.create_dataset('mo_occ', data=subsys.mo_occ)
            sub_mol.create_dataset('mo_energy', data=subsys.mo_energy)

    return True

def read_chkfile(subsys, filename=None):
    if filename is None:
        if subsys.filename is None:
            return False
        filename = os.path.splitext(subsys.filename)[0] + '.hdf5'
    if os.path.isfile(filename):
        try:
            with h5py.File(filename, 'r') as h5_file:
                subsys_coeff = h5_file['subsystem/mo_coeff']
                if subsys.mol.nao == subsys_coeff.shape[1]:
                    subsys.mf_obj.mo_coeff = subsys_coeff[:]
                    subsys_occ = h5_file['subsystem/mo_occ']
                    subsys.mf_obj.mo_occ = subsys_occ[:]
                    subsys_energy = h5_file['subsystem/mo_energy']
                    subsys.mf_obj.mo_energy = subsys_energy[:]
                    return True
                logger.warn(subsys.mol, "chkfile improperly formatted")
                return False
        except TypeError:
            logger.warn(subsys.mol, "chkfile improperly formatted")
            return False
    logger.warn(subsys.mol, "chkfile NOT found")
    return False

def save_density(subsys, filename=None, density=None):
    if filename is None:
        filename = subsys.filename
    if density is None:
        density = subsys.make_rdm1()
    
    logger.note(subsys.mol, 'writing cubegen density at:%s', filename)

    if density.ndim > 2:
        cube_fn = os.path.splitext(filename)[0] + '_alpha.cube'
        cubegen.density(subsys.mol, cube_fn, density[0])
        cube_fn = os.path.splitext(filename)[0] + '_beta.cube'
        cubegen.density(subsys.mol, cube_fn, density[1])
    else:
        cube_fn = os.path.splitext(filename)[0] + '.cube'
        cubegen.density(subsys.mol, cube_fn, density)

def save_spin_density(subsys, filename=None, density=None):
    if filename is None:
        filename = subsys.filename
    if density is None:
        density = subsys.make_rdm1()
    
    logger.note(subsys.mol, 'writing cubegen spin-density at:%s', filename)

    if density.ndim > 2:
        cube_fn = os.path.splitext(filename)[0] + '_spin_dm.cube'
        cubegen.density(subsys.mol, cube_fn, density[0]-density[1])
        return True
    else:
        logger.warn(subsys.mol, 'cannot write spin density of closed shell density', filename)
        return False

def save_orbitals(subsys, filename=None, mo_coeff=None, mo_occ=None, mo_energy=None):

    if filename is None:
        filename = subsys.filename
    if mo_occ is None:
         mo_occ = subsys.qm_obj.mo_occ
    if mo_coeff is None:
         mo_coeff = subsys.qm_obj.mo_coeff
    if mo_energy is None:
         mo_energy = subsys.qm_obj.mo_energy
    logger.note(subsys.mol, 'writing molden orbitals at:%s', filename)

    if mo_coeff.ndim > 2:
        molden_fn = os.path.splitext(filename)[0] + '_alpha.molden'
        with open(molden_fn, 'w') as fin:
            molden.header(subsys.mol, fin)
            molden.orbital_coeff(subsys.mol, fin, mo_coeff[0],
                                 spin='Alpha', ene=mo_energy[0], occ=mo_occ[0])
        molden_fn = os.path.splitext(filename)[0] + '_beta.molden'
        with open(molden_fn, 'w') as fin:
            molden.header(subsys.mol, fin)
            molden.orbital_coeff(subsys.mol, fin, mo_coeff[1],
                                 spin='Beta', ene=mo_energy[1], occ=mo_occ[1])

    else:
        molden_fn = os.path.splitext(filename)[0] + '.molden'
        with open(molden_fn, 'w') as fin:
            molden.header(subsys.mol, fin)
            molden.orbital_coeff(subsys.mol, fin, mo_coeff,
                                 ene=mo_energy, occ=mo_occ)

class QMSubsystem:

    def __init__(self, qm_obj, init_guess='sup'):
        self.qm_obj = qm_obj
        self.mf_obj = None
        if issubclass(type(self.qm_obj), scf.hf.SCF):
            self.mf_obj = self.qm_obj
        elif (type(self.qm_obj) is str):
            print ('external method')
        else:
            self.mf_obj = self.qm_obj._scf
        self.mol = self.mf_obj.mol
        self.conv_tol = self.mf_obj.conv_tol
        self.max_cycle = self.mf_obj.max_cycle
        self.subcycle = 1
        self.proj_pot = 0.
        self.emb_pot = 0.
        self.emb_fock = 0.
        self.init_dmat = None
        self.init_guess = init_guess
        self.subsys_get_fock = self.mf_obj.get_fock
        self.mf_obj.get_hcore = lambda *args: scf.hf.get_hcore(self.mol) + self.emb_pot + self.proj_pot

    def use_emb_fock(self):
        #Overwrites the get_fock method to just return the embedded fock. This speeds up execution to avoid calculating multiple fock matrices twice.
        self.mf_obj.get_fock = lambda *args, **kwargs: self.emb_fock + self.proj_pot

    def reset_fock(self):
        #Resets the get_fock method to be just for the subsystem not an embedded fock matrix.
        self.mf_obj.get_fock = self.subsys_get_fock

    def relax(self, subcycle=None):
        if subcycle is None:
            subcycle = self.subcycle
        prev_max_cycle = self.qm_obj.max_cycle
        prev_verbose = self.qm_obj.verbose
        self.qm_obj.verbose = 0
        self.qm_obj.max_cycle = subcycle
        out_val = self.qm_obj.scf(dm0=self.make_rdm1())
        self.qm_obj.verbose = prev_verbose
        self.qm_obj.max_cycle = prev_max_cycle
        return out_val


    def init_den(self, sup_dmat=None):
        if self.init_guess is 'sup' and sup_dmat is not None:
            self.init_dmat = sup_dmat
        else:
            self.init_dmat = self.mf_obj.get_init_guess()


    def make_rdm1(self):
        if self.mf_obj.mo_coeff is None:
            return self.init_dmat
        return self.mf_obj.make_rdm1()

    def energy_tot(self):
        return self.qm_obj.energy_tot()

    def kernel(self):
        if issubclass(type(self.qm_obj), scf.hf.SCF):
            return self.qm_obj.scf(dm0=self.make_rdm1())
        elif (type(self.qm_obj) is str):
            print ('external method')
        else:
            print ('init')
            self.mf_obj.kernel()
            #initialize the qm object again with the new mf_obj
