# A method to define a cluster supersystem
# Daniel Graham

import os
from qsome import supersystem
from pyscf import gto, scf, dft

import numpy as np
import h5py

class ClusterSuperSystem(supersystem.SuperSystem):

    def __init__(self, subsystems, ct_method, proj_oper='huz', filename=None,
                 ft_cycles=100, ft_conv=1e-8, ft_grad=1e-8, ft_diis=1, 
                 ft_setfermi=None, ft_initguess=None, ft_updatefock=0, 
                 cycles=100, conv=1e-8, grad=1e-8, damp=0, shift=0, 
                 smearsigma=0, initguess=None, includeghost=False, 
                 grid_level=4, verbose=3, analysis=False, debug=False):

        self.subsystems = subsystems
        self.ct_method = ct_method
        self.proj_oper = proj_oper

        if filename is None:
            filename = os.getcwd() + '/temp.inp'
        self.filename = filename

        self.chk_filename = os.path.splitext(self.filename)[0] + '.hdf5'

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
        self.grid_level = grid_level
        self.verbose = verbose
        self.analysis = analysis
        self.debug = debug

        # These are also stored in the pyscf object, but if we do a custom diagonalizaiton, they must be stored separately.
        # Actually, could just modify the pyscf object attributes...
        # Actually will not use pyscf attributes. Want to store consistently in the same way, so always store alpha and beta. Makes everything less complicated.
        self.mo_coeff = None
        self.mo_occ = None
        self.mo_energy = None

        self.concat_mols()
        self.gen_sub2sup()
        self.init_ct_scf()
        self.init_density()

        self.fock = [None, None]
        self.hcore = self.ct_scf.get_hcore()
        self.proj_pot = [[None, None] for i in range(len(subsystems))]
        self.smat = self.ct_scf.get_ovlp()

        self.update_fock()
        self.ft_fermi = [[0., 0.] for i in range(len(subsystems))]

    def gen_sub2sup(self):
        nao = np.array([ self.subsystems[i].mol.nao_nr() for i in range(len(self.subsystems)) ])
        nssl = [ None for i in range(len(self.subsystems)) ]

        for i in range(len(self.subsystems)):
            nssl[i] = np.zeros(self.subsystems[i].mol.natm, dtype=int)
            for j in range(self.subsystems[i].mol.natm):
                ib = np.where(self.subsystems[i].mol._bas.transpose()[0] == j)[0].min()
                ie = np.where(self.subsystems[i].mol._bas.transpose()[0] == j)[0].max()
                ir = self.subsystems[i].mol.nao_nr_range(ib, ie + 1)
                ir = ir[1] - ir[0]
                nssl[i][j] = ir

            if nssl[i].sum() != self.subsystems[i].mol.nao_nr():
                print ('ERROR: naos not equal!') # should be a logged statement.

        mAB = self.mol

        nsl = np.zeros(mAB.natm, dtype=int)
        for i in range(mAB.natm):
            ib = np.where(mAB._bas.transpose()[0] == i)[0].min()
            ie = np.where(mAB._bas.transpose()[0] == i)[0].max()
            ir = mAB.nao_nr_range(ib, ie + 1)
            ir = ir[1] - ir[0]
            nsl[i] = ir

        if nsl.sum() != mAB.nao_nr():
            print ('ERROR: naos not equal!') # should be a logged statement

        sub2sup = [ None for i in range(len(self.subsystems)) ]
        for i in range(len(self.subsystems)):
            sub2sup[i] = np.zeros(nao[i], dtype=int)
            for a in range(self.subsystems[i].mol.natm):
                match = False
                for b in range(mAB.natm):
                    c1 = self.subsystems[i].mol.atom_coord(a)
                    c2 = mAB.atom_coord(b)
                    d = np.dot(c1 - c2, c1 - c2)
                    if d < 0.001:
                        match = True
                        ia = nssl[i][0:a].sum()
                        ja = ia + nssl[i][a]
                        ib = nsl[0:b].sum()
                        jb = ib + nsl[b]
                        sub2sup[i][ia:ja] = range(ib, jb)

                if not match:
                    print ('ERROR: I did not find an atom match!') # should be logged

        self.sub2sup = sub2sup

    def init_ct_scf(self):
        if self.ct_method[0] == 'u':
            if self.ct_method[1:] == 'hf':
                scf_obj = scf.UHF(self.mol) 
            else:
                scf_obj = scf.UKS(self.mol)
                scf_obj.xc = self.ct_method[1:]
                scf_obj.small_rho_cutoff = 1e-20 #this prevents pruning. Also slows down code. Can probably remove and use default in pyscf (1e-7)
        elif self.ct_method[:2] == 'ro':
            if self.ct_method[2:] == 'hf':
                scf_obj = scf.ROHF(self.mol) 
            else:
                scf_obj = scf.ROKS(self.mol)
                scf_obj.xc = self.ct_method[2:]
                scf_obj.small_rho_cutoff = 1e-20 #this prevents pruning. Also slows down code. Can probably remove and use default in pyscf (1e-7)
        else:
            if self.ct_method == 'hf' or self.ct_method[1:] == 'hf':
               scf_obj = scf.RHF(self.mol) 
            else:
                scf_obj = scf.RKS(self.mol)
                scf_obj.xc = self.ct_method
                if self.ct_method[0] == 'r':
                    scf_obj.xc = self.ct_method[1:]
                scf_obj.small_rho_cutoff = 1e-20 #this prevents pruning. Also slows down code. Can probably remove and use default in pyscf (1e-7)

        self.ct_scf = scf_obj

    def init_density(self):

        # Finish reading from chkfile
        # This method could be cleaned up, but it works.
        print ("".center(80,'*'))
        print ("  Generate Initial System Densities  ".center(80))
        print ("".center(80,'*'))
        # create grid.
        self.grids = dft.gen_grid.Grids(self.mol)
        self.grids.level = self.grid_level
        self.grids.build()
        self.ct_scf.grids = self.grids
        self.dmat = [None, None]

        #Initiate all subsystem densities.
        sup_calc = any([subsystem.initguess == 'supmol' or (subsystem.initguess is None and (self.ft_initguess == 'supmol' or self.initguess == 'supmol')) for subsystem in self.subsystems])
        if sup_calc:
            self.supermolecular_energy()
        s2s = self.sub2sup
        readchk_init = any([subsystem.initguess == 'readchk' or (subsystem.initguess is None and (self.ft_initguess == 'readchk' or self.initguess == 'readchk')) for subsystem in self.subsystems])
        if readchk_init:
            is_chkfile = self.read_chkfile()
        for i in range(len(self.subsystems)):
            subsystem = self.subsystems[i]
            subsystem.env_scf.grids = self.grids
            if subsystem.initguess is None:
                if self.ft_initguess == 'supmol':
                    subsystem.dmat[0] = self.dmat[0][np.ix_(s2s[i], s2s[i])]
                    subsystem.dmat[1] = self.dmat[1][np.ix_(s2s[i], s2s[i])]

                elif self.ft_initguess == 'readchk':
                    if is_chkfile:
                        if (not subsystem.env_mo_coeff is None) and (not subsystem.env_mo_occ):
                            subsystem.dmat[0] = np.dot((subsystem.env_mo_coeff[0] * subsystem.env_mo_occ[0]), subsystem.env_mo_coeff[0].transpose().conjugate())
                            subsystem.dmat[1] = np.dot((subsystem.env_mo_coeff[1] * subsystem.env_mo_occ[1]), subsystem.env_mo_coeff[1].transpose().conjugate())
                        elif (not self.mo_coeff is None) and (not self.mo_occ):
                            sup_dmat[0] = np.dot((self.mo_coeff[0] * self.mo_occ[0]), self.mo_occ[0].transpose().conjugate())
                            sup_dmat[1] = np.dot((self.mo_coeff[1] * self.mo_occ[1]), self.mo_occ[1].transpose().conjugate())
                            subsystem.dmat[0] = sup_dmat[0][np.ix_(s2s[i], s2s[i])]
                            subsystem.dmat[1] = sup_dmat[1][np.ix_(s2s[i], s2s[i])]
                        else:
                            temp_dmat = self.ct_scf.get_init_guess()
                            if temp_dmat.ndim == 2:  #Temp dmat is only one dimensional
                                t_d = [temp_dmat.copy()/2., temp_dmat.copy()/2.]
                                temp_dmat = t_d
                            subsystem.dmat[0] = temp_dmat[0][np.ix_(s2s[i], s2s[i])]
                            subsystem.dmat[1] = temp_dmat[1][np.ix_(s2s[i], s2s[i])]
                    else:
                        temp_dmat = self.ct_scf.get_init_guess()
                        if temp_dmat.ndim == 2:  #Temp dmat is only one dimensional
                            t_d = [temp_dmat.copy()/2., temp_dmat.copy()/2.]
                            temp_dmat = t_d
                        subsystem.dmat[0] = temp_dmat[0][np.ix_(s2s[i], s2s[i])]
                        subsystem.dmat[1] = temp_dmat[1][np.ix_(s2s[i], s2s[i])]
                else:
                    if self.ft_initguess != None:
                        temp_dmat = self.ct_scf.get_init_guess(key=self.ft_initguess)
                    else:
                        temp_dmat = self.ct_scf.get_init_guess()
                    if temp_dmat.ndim == 2:  #Temp dmat is only one dimensional
                        t_d = [temp_dmat.copy()/2., temp_dmat.copy()/2.]
                        temp_dmat = t_d
                    subsystem.dmat[0] = temp_dmat[0][np.ix_(s2s[i], s2s[i])]
                    subsystem.dmat[1] = temp_dmat[1][np.ix_(s2s[i], s2s[i])]

            elif subsystem.initguess == 'supmol': 
                subsystem.dmat[0] = self.dmat[0][np.ix_(s2s[i], s2s[i])]
                subsystem.dmat[1] = self.dmat[1][np.ix_(s2s[i], s2s[i])]
            elif subsystem.initguess == 'readchk':
                if is_chkfile:
                    if (not subsystem.env_mo_coeff is None) and (not subsystem.env_mo_occ is None):
                        subsystem.dmat[0] = np.dot((subsystem.env_mo_coeff[0] * subsystem.env_mo_occ[0]), subsystem.env_mo_coeff[0].transpose().conjugate())
                        subsystem.dmat[1] = np.dot((subsystem.env_mo_coeff[1] * subsystem.env_mo_occ[1]), subsystem.env_mo_coeff[1].transpose().conjugate())
                    elif (not self.mo_coeff is None) and (not self.mo_occ is None):
                        sup_dmat = [None, None]
                        sup_dmat[0] = np.dot((self.mo_coeff[0] * self.mo_occ[0]), self.mo_coeff[0].transpose().conjugate())
                        sup_dmat[1] = np.dot((self.mo_coeff[1] * self.mo_occ[1]), self.mo_coeff[1].transpose().conjugate())
                        subsystem.dmat[0] = sup_dmat[0][np.ix_(s2s[i], s2s[i])]
                        subsystem.dmat[1] = sup_dmat[1][np.ix_(s2s[i], s2s[i])]
                    else:
                        temp_dmat = self.ct_scf.get_init_guess()
                        if temp_dmat.ndim == 2:  #Temp dmat is only one dimensional
                            t_d = [temp_dmat.copy()/2., temp_dmat.copy()/2.]
                            temp_dmat = t_d
                        subsystem.dmat[0] = temp_dmat[0][np.ix_(s2s[i], s2s[i])]
                        subsystem.dmat[1] = temp_dmat[1][np.ix_(s2s[i], s2s[i])]
            else:
                if subsystem.initguess != None:
                    temp_dmat = self.ct_scf.get_init_guess(key=subsystem.initguess)
                else:
                    temp_dmat = self.ct_scf.get_init_guess()

                if temp_dmat.ndim == 2:  #Temp dmat is only one dimensional
                    t_d = [temp_dmat.copy()/2., temp_dmat.copy()/2.]
                    temp_dmat = t_d
                subsystem.dmat[0] = temp_dmat[0][np.ix_(s2s[i], s2s[i])]
                subsystem.dmat[1] = temp_dmat[1][np.ix_(s2s[i], s2s[i])]

        #initialize full system density
        if not sup_calc:
            if self.initguess == 'readchk':
                if is_chkfile:
                    if (not self.mo_coeff is None) and (not self.mo_occ is None):
                        self.dmat[0] = np.dot((self.mo_coeff[0] * self.mo_occ[0]), self.mo_coeff[0].transpose().conjugate())
                        self.dmat[1] = np.dot((self.mo_coeff[1] * self.mo_occ[1]), self.mo_coeff[1].transpose().conjugate())
                    else:
                        temp_dmat = self.ct_scf.get_init_guess()
                        if temp_dmat.ndim == 2:  #Temp dmat is only one dimensional
                            t_d = [temp_dmat.copy()/2., temp_dmat.copy()/2.]
                            self.dmat = t_d
            
            else:
                if self.initguess != None:
                    self.dmat = self.ct_scf.get_init_guess(key=self.initguess)
                else:
                    self.dmat = self.ct_scf.get_init_guess()

                if self.dmat.ndim == 2:  #Temp dmat is only one dimensional
                    t_d = [self.dmat.copy()/2., self.dmat.copy()/2.]
                    self.dmat = t_d
        print ("".center(80,'*'))
    

    def concat_mols(self):

        # this works but can be done MUCH better.
        self.mol = gto.Mole()
        self.mol.basis = {}
        atm = []
        nghost = 0
        self.mol.charge = 0
        self.mol.spin = 0
        for i in range(len(self.subsystems)):
            subsystem = self.subsystems[i]
            self.mol.charge += subsystem.mol.charge
            self.mol.spin += subsystem.mol.spin
            for j in range(subsystem.mol.natm):
                if 'ghost' in subsystem.mol.atom_symbol(j).lower():
                    if self.includeghost:
                        nghost += 1
                        ghost_name = subsystem.mol.atom_symbol(j).split(':')[0] + f':{nghost}'
                        atm.append([ghost_name, subsystem.mol.atom_coord(j)])
                        self.mol.basis.update({ghost_name: subsystem.mol.basis[subsystem.mol.atom_symbol(j)]})
                else:
                    if i > 0:
                        atom_name = subsystem.mol.atom_symbol(j) + ':' + str(i)
                    else:
                        atom_name = subsystem.mol.atom_symbol(j)
                    atm.append([atom_name, subsystem.mol.atom_coord(j)])
                    self.mol.basis.update({atom_name: subsystem.mol.basis[subsystem.mol.atom_symbol(j)]})

        self.mol.atom = atm
        self.mol.verbose = self.verbose
        self.mol.unit = 'bohr' # atom_coord is always stored in bohr for some reason. Trust me this is right.
        self.mol.build(dump_input=False)

    def supermolecular_energy(self):

        #init_dmat = self.ct_scf.get_init_guess()
        print ("".center(80,'*'))
        print("  SuperSystem Calculation  ".center(80))
        print ("".center(80,'*'))
        self.ct_scf.scf()
        self.dmat = self.ct_scf.make_rdm1()
        if self.dmat.ndim == 2: #Always store as alpha and beta, even if closed shell. Makes calculations easier.
            t_d = [self.dmat.copy()/2., self.dmat.copy()/2.]
            self.dmat = t_d
            self.mo_coeff = [self.ct_scf.mo_coeff, self.ct_scf.mo_coeff]
            self.mo_occ = [self.ct_scf.mo_occ/2, self.ct_scf.mo_occ/2]
            self.mo_energy = [self.ct_scf.mo_energy, self.ct_scf.mo_energy]
        
        self.save_chkfile()
        print("".center(80,'*'))
        return self.ct_scf.energy_tot()

    def env_in_env_energy(self):
        pass

    def get_proj_op(self):
        pass

    def get_embedding_pot(self):
        pass

    def update_fock(self):
        self.fock = [np.copy(self.hcore), np.copy(self.hcore)]

        # Optimization: Rather than recalculate the full V, only calculate the V for densities which changed. 
        # get 2e matrix
        nS = self.mol.nao_nr()
        dm = [np.zeros((nS, nS)), np.zeros((nS, nS))]
        for i in range(len(self.subsystems)):
            dm[0][np.ix_(self.sub2sup[i], self.sub2sup[i])] += self.subsystems[i].dmat[0]
            dm[1][np.ix_(self.sub2sup[i], self.sub2sup[i])] += self.subsystems[i].dmat[1]

        if self.ct_method[0] == 'u' or self.ct_method[:2] == 'ro':
            V = self.ct_scf.get_veff(mol=self.mol, dm=dm)
            V_a = V[0]
            V_b = V[1]
        else:
            V_a = self.ct_scf.get_veff(mol=self.mol, dm=dm[0])
            V_b = self.ct_scf.get_veff(mol=self.mol, dm=dm[1])
        self.fock[0] += V_a
        self.fock[1] += V_b

    def update_proj_op(self):
        # currently updates both at once. Can easily modify to only update one subsystem, however I don't think it will improve the speed.
        s2s = self.sub2sup
        for i in range(len(self.subsystems)):
            A = i
            nA = self.subsystems[A].mol.nao_nr()
            SAA = self.smat[np.ix_(s2s[A], s2s[A])]
            POp = [np.zeros((nA, nA)), np.zeros((nA, nA))]

            # cycle over all other subsystems
            for B in range(len(self.subsystems)):
                if B==A: continue

                SAB = self.smat[np.ix_(s2s[A], s2s[B])]
                SBA = self.smat[np.ix_(s2s[B], s2s[A])]

                # get mu-parameter projection operator
                if isinstance(self.proj_oper, int) or isinstance(self.proj_oper, float):
                    POp[0] += self.proj_oper * np.dot( SAB, np.dot( self.subsystems[B].dmat[0], SBA ))
                    POp[1] += self.proj_oper * np.dot( SAB, np.dot( self.subsystems[B].dmat[1], SBA ))

                elif self.proj_oper in ('huzinaga', 'huz'):
                    FAB = [None, None]
                    FAB[0] = self.fock[0][np.ix_(s2s[A], s2s[B])]
                    FAB[1] = self.fock[1][np.ix_(s2s[A], s2s[B])]
                    FDS = [None, None]
                    FDS[0] = np.dot( FAB[0], np.dot( self.subsystems[B].dmat[0], SBA ))
                    FDS[1] = np.dot( FAB[1], np.dot( self.subsystems[B].dmat[1], SBA ))
                    POp[0] += - 0.5 * ( FDS[0] + FDS[0].transpose() ) #May not need 0.5 cause divide into alpha and beta
                    POp[1] += - 0.5 * ( FDS[0] + FDS[0].transpose() )

                elif self.proj_oper in ('huzinagafermi', 'huzfermi'):
                    FAB = [None, None]
                    FAB[0] = self.fock[0][np.ix_(s2s[A], s2s[B])]
                    FAB[1] = self.fock[1][np.ix_(s2s[A], s2s[B])]
                    #The max of the fermi energy
                    efermi = [None, None]
                    if self.ft_setfermi is None:
                        efermi[0] = max([fermi[0] for fermi in self.ft_fermi])
                        efermi[1] = max([fermi[1] for fermi in self.ft_fermi])
                    else:
                        efermi[0] = self.ft_setfermi
                        efermi[1] = self.ft_setfermi #Allow for two set fermi, one for a and one for b

                    FAB[0] -= SAB * efermi
                    FAB[1] -= SAB * efermi #could probably specify fermi for each alpha or beta electron.

                    FDS = [None, None]
                    FDS[0] = np.dot( FAB[0], np.dot( self.subsystems[B].dmat[0], SBA ))
                    FDS[0] = np.dot( FAB[1], np.dot( self.subsystems[B].dmat[1], SBA ))
                    POp[0] += - 0.5 * ( FDS[0] + FDS[0].transpose() ) #may not need 0.5
                    POp[1] += - 0.5 * ( FDS[1] + FDS[1].transpose() )

            self.proj_pot[i] = POp.copy()

    def read_chkfile(self):
        if os.path.isfile(self.chk_filename):
            with h5py.File(self.chk_filename, 'r') as hf:
                supsys_coeff = hf['supersystem/mo_coeff']
                if supsys_coeff.shape is None:
                    print("SuperSystem MO Coeff Empty".center(80))
                else:
                    self.mo_coeff = supsys_coeff[:]

                supsys_occ = hf['supersystem/mo_occ']
                if supsys_occ.shape is None:
                    print("SuperSystem Occupation Empty".center(80))
                else:
                    self.mo_occ = supsys_occ[:]
                supsys_energy = hf['supersystem/mo_energy']
                if supsys_energy.shape is None:
                    print("SuperSystem MO Energies Empty".center(80))
                else:
                    self.mo_energy = supsys_energy[:]

                for i in range(len(self.subsystems)):
                    subsystem = self.subsystems[i]
                    subsys_coeff = hf[f'subsystem:{i}/mo_coeff']
                    if subsys_coeff.shape is None:
                        print(f"Subsystem:{i} MO Coeff Empty".center(80))
                    else:
                        subsystem.env_mo_coeff = subsys_coeff[:]
                    subsys_occ = hf[f'subsystem:{i}/mo_occ']
                    if subsys_occ.shape is None:
                        print(f"Subsystem:{i} Occupation Empty".center(80))
                    else:
                        subsystem.env_mo_occ = subsys_occ[:]
                    subsys_energy = hf[f'subsystem:{i}/mo_energy']
                    if subsys_energy.shape is None:
                        print(f"Subsystem:{i} MO Energies Empty".center(80))
                    else:
                        subsystem.env_mo_energy = subsys_energy[:]
            return True 
        else:
            print ("chkfile NOT found".center(80))
            return False

    def save_chkfile(self):
        # current plan is to save mo_coefficients, occupation vector, and energies.
        # becasue of how h5py works we need to check if none and save as the correct filetype (f)
        
        # check if file exists. 
        if os.path.isfile(self.chk_filename):
            with h5py.File(self.chk_filename, 'r+') as hf:
                supsys_coeff = hf['supersystem/mo_coeff']
                if supsys_coeff.shape is None:
                    if not self.mo_coeff is None:
                        del supsys_coeff
                        hf.create_dataset('supersystem/mo_coeff', data=self.mo_coeff) 
                else:
                    if not self.mo_coeff is None:
                        supsys_coeff[...] = self.mo_coeff
                    else:
                        del supsys_coeff
                        hf.create_dataset('supersystem/mo_coeff', dtype='f') 

                supsys_occ = hf['supersystem/mo_occ']
                if supsys_occ.shape is None:
                    if not self.mo_occ is None:
                        del supsys_occ
                        hf.create_dataset('supersystem/mo_occ', data=self.mo_occ)
                else:
                    if not self.mo_occ is None:
                        supsys_occ[...] = self.mo_occ
                    else:
                        del supsys_occ
                        hf.create_dataset('supersystem/mo_occ', dtype='f') 
                supsys_energy = hf['supersystem/mo_energy']
                if supsys_energy.shape is None:
                    if not self.mo_energy is None:
                        del supsys_energy
                        hf.create_dataset('supersystem/mo_energy', data=self.mo_energy)
                else:
                    if not self.mo_energy is None:
                        supsys_energy[...] = self.mo_energy
                    else:
                        del supsys_energy
                        hf.create_dataset('supersystem/mo_energy', dtype='f') 
                for i in range(len(self.subsystems)):
                    subsystem = self.subsystems[i]
                    subsys_coeff = hf[f'subsystem:{i}/mo_coeff']
                    if subsys_coeff.shape is None:
                        if not subsystem.env_mo_coeff is None:
                            del subsys_coeff
                            hf.create_dataset(f'subsystem:{i}/mo_coeff', data=subsystem.env_mo_coeff)
                    else:
                        if not subsystem.env_mo_coeff is None:
                            subsys_coeff[...] = subsystem.env_mo_coeff
                        else:
                            del subsys_coeff
                            hf.create_dataset(f'subsystem:{i}/mo_coeff', dtype='f')
                    subsys_occ = hf[f'subsystem:{i}/mo_occ']
                    if subsys_occ.shape is None:
                        if not subsystem.env_mo_occ is None:
                            del subsys_occ
                            hf.create_dataset(f'subsystem:{i}/mo_occ', data=subsystem.env_mo_occ)
                    else:
                        if not subsystem.env_mo_occ is None:
                            subsys_occ[...] = subsystem.env_mo_occ
                        else:
                            del subsys_occ
                            hf.create_dataset(f'subsystem:{i}/mo_occ', dtype='f')
                    subsys_energy = hf[f'subsystem:{i}/mo_energy']
                    if subsys_energy.shape is None:
                        if not subsystem.env_mo_energy is None:
                            del subsys_energy
                            hf.create_dataset(f'subsystem:{i}/mo_energy', data=subsystem.env_mo_energy)
                    else:
                        if not subsystem.env_mo_energy is None:
                            subsys_energy[...] = subsystem.env_mo_energy
                        else:
                            del subsys_energy
                            hf.create_dataset(f'subsystem:{i}/mo_energy', dtype='f')
        else:
            with h5py.File(self.chk_filename, 'w') as hf:
                hf.create_dataset("embedding_chkfile", data=True)
                sup_mol = hf.create_group('supersystem')
                if self.mo_coeff is None:
                    sup_mol.create_dataset('mo_coeff', dtype="f")
                else:
                    sup_mol.create_dataset('mo_coeff', data=self.mo_coeff)
                if self.mo_occ is None:
                    sup_mol.create_dataset('mo_occ', dtype="f")
                else:
                    sup_mol.create_dataset('mo_occ', data=self.mo_occ)

                if self.mo_energy is None:
                    sup_mol.create_dataset('mo_energy', dtype="f")
                else:
                    sup_mol.create_dataset('mo_energy', data=self.mo_energy)

                for i in range(len(self.subsystems)):
                    subsystem = self.subsystems[i]
                    sub_sys_data = hf.create_group(f'subsystem:{i}')
                    if subsystem.env_mo_coeff is None:
                        sub_sys_data.create_dataset('mo_coeff', dtype="f")
                    else:
                        sub_sys_data.create_dataset('mo_coeff', data=subsystem.env_mo_coeff)
                    if subsystem.env_mo_occ is None:
                        sub_sys_data.create_dataset('mo_occ', dtype="f")
                    else:
                        sub_sys_data.create_dataset('mo_occ', data=subsystem.env_mo_occ)
                    if subsystem.env_mo_energy is None:
                        sub_sys_data.create_dataset('mo_energy', dtype="f")
                    else:
                        sub_sys_data.create_dataset('mo_energy', data=subsystem.env_mo_energy)

    def freeze_and_thaw(self):
        #Optimization: rather than recalculate vA use the existing fock and subtract out the block that is double counted.

        print("".center(80, '*'))
        print("Freeze-and-Thaw".center(80))
        print("".center(80, '*'))
         
        s2s = self.sub2sup
        ft_err = 1.
        ft_iter = 0 
        while((ft_err > self.ft_conv) and (ft_iter < self.ft_cycles)):
            # cycle over subsystems
            ft_err = 0 
            ft_iter += 1
            for i in range(len(self.subsystems)):
                subsystem = self.subsystems[i]
                if not subsystem.frozen:
                    if self.fock_update >= i:
                        self.update_fock()

                    #this will slow down calculation. 
                    sub_old_e = subsystem.get_env_energy()
                    sub_old_dm = subsystem.dmat.copy()

                    self.update_proj_pot() #could use i as input and only get for that sub.
                    FAA[0] = self.fock[0][np.ix_(s2s[i], s2s[i])]
                    FAA[1] = self.fock[1][np.ix_(s2s[i], s2s[i])]
             
                    SAA = self.smat[np.ix_(s2s[i], s2s[i])]
                    #I don't think this changes. Could probably set in the initialize.
                    sub_hcore = copy(self.hcore[np.ix_(s2s[i], s2s[i])])
                    subsystem.env_scf.get_hcore = lambda *args: sub_hcore

                    subsystem.update_fock()
                    froz_veff = [None, None]
                    froz_veff[0] = (FAA[0] - sub_hcore - subsystem.env_vA[0])
                    froz_veff[1] = (FAA[0] - sub_hcore - subsystem.emb_vA[1])
                    subsystem.update_emb_pot(froz_veff)
                    subsystem.update_proj_pot(self.proj_pot[i])
                    # diagonalize here.
                    subsystem.diagonalize()
                    # save to file. could be done in larger cycles.
                    self.save_chkfile()
                    ddm = sp.linalg.norm(subsystem.dmat[0] - sub_old_dm[0])
                    ddm += sp.linalg.norm(subsystem.dmat[1] - sub_old_dm[1])
                    proj_e = np.trace(np.dot(subsystem.dmat[0], self.proj_pot[i][0]))
                    proj_e += np.trace(np.dot(subsystem.dmat[1], self.proj_pot[i][1]))
                    ft_err += ddm

                    self.ft_fermi[i] = subsystem.fermi
                    #This will slow down execution.
                    sub_new_e = subsystem.get_env_energy()
                    dE = abs(sub_old_e - sub_new_e)

                    # print output to console.
                    print(f"iter: {ft_iter:>3d}:{subsystem.sub_num:<2d}  |dE|: {dE:12.6e}   |ddm|: {ddm:12.6e}   |Tr[DP]|: {proj_e:12.6e}")

        print("".center(80))
        if(ft_err > self.ft_conv):
            print("".center(80))
            print" Freeze-and-Thaw NOT converged".center(80))

        # print subsystem energies 
        for subsystem in self.subsystems:
            print(f"Subsystem {subsystem.sub_num} Energy: {subsystem.get_emb_energy():12.8f}")
        print("".center(80))
        print("".center(80, '*'))
