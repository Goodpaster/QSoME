# A method to define a cluster supersystem
# Daniel Graham

import os
from qsome import supersystem, custom_pyscf_methods
from pyscf import gto, scf, dft, lib

from pyscf.tools import cubegen

import functools
import time

import numpy as np
import scipy as sp
import h5py

def time_method(function_name=None):
    def real_decorator(func):
        @functools.wraps(func)
        def wrapper_time_method(*args, **kwargs):
            ts = time.time()
            result = func(*args, **kwargs)
            te = time.time()
            if function_name is None:
                name = func.__name__.upper()
            else:
                name = function_name
            elapsed_t = (te - ts)
            print( f'{name:40} {elapsed_t:.4f}s')
            return result
        return wrapper_time_method 
    return real_decorator

class ClusterSuperSystem(supersystem.SuperSystem):

    def __init__(self, subsystems, ct_method, proj_oper='huz', filename=None,
                 ft_cycles=100, ft_conv=1e-8, ft_grad=None, ft_diis=1, 
                 ft_setfermi=None, ft_initguess=None, ft_updatefock=0,
                 ft_writeorbs=False, cycles=100, conv=1e-9, grad=None, 
                 damp=0, shift=0, smearsigma=0, initguess=None, 
                 grid_level=4, verbose=3, analysis=False, debug=False, 
                 rhocutoff=1e-7):

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
        self.ft_writeorbs = ft_writeorbs


        self.ft_setfermi = ft_setfermi
        self.ft_initguess = ft_initguess
        self.ft_updatefock = ft_updatefock

        # charge transfer settings
        self.cycles = cycles
        self.conv = conv
        self.grad = grad
        self.rho_cutoff = rhocutoff
        self.damp = damp
        self.shift = shift
        self.smearsigma = smearsigma
        self.initguess = initguess

        # general system settings
        self.grid_level = grid_level
        self.verbose = verbose
        self.analysis = analysis #provide a more detailed analysis at higher computational cost
        self.debug = debug

        # These are also stored in the pyscf object, but if we do a custom diagonalizaiton, they must be stored separately.
        # Actually, could just modify the pyscf object attributes...
        # Actually will not use pyscf attributes. Want to store consistently in the same way, so always store alpha and beta. Makes everything less complicated.

        self.freeze_and_thaw_conv = False
        self.mol = self.concat_mols()
        self.gen_sub2sup()
        self.init_ct_scf()
        self.init_density()

        #There are other diis methods but these don't work with out method due to subsystem projection.
        if ft_diis == 0:
            self.ft_diis = None
        #The scf diis methods do not work for a system where there is a projection potential separating subsystems. DIIS could be altered to allow for this to optimize.
        #elif ft_diis == 1:
        #    self.ft_diis = [scf.diis.DIIS(mf=self.ct_scf), scf.diis.DIIS(self.ct_scf)]
        #elif ft_diis == 2:
        #    self.ft_diis = [scf.diis.EDIIS(), scf.diis.EDIIS()]
        #elif ft_diis == 3:
        #    self.ft_diis = [scf.diis.ADIIS(), scf.diis.ADIIS()]
        else:
            self.ft_diis = [lib.diis.DIIS(), lib.diis.DIIS()]


        self.update_fock(diis=False)
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
                scf_obj.small_rho_cutoff = self.rho_cutoff #this prevents pruning. Also slows down code. Can probably remove and use default in pyscf (1e-7)
        elif self.ct_method[:2] == 'ro':
            if self.ct_method[2:] == 'hf':
                scf_obj = scf.ROHF(self.mol) 
            else:
                scf_obj = scf.ROKS(self.mol)
                scf_obj.xc = self.ct_method[2:]
                scf_obj.small_rho_cutoff = self.rho_cutoff #this prevents pruning. Also slows down code. Can probably remove and use default in pyscf (1e-7)
        else:
            if self.ct_method == 'hf' or self.ct_method[1:] == 'hf':
               scf_obj = scf.RHF(self.mol) 
            else:
                scf_obj = scf.RKS(self.mol)
                scf_obj.xc = self.ct_method
                if self.ct_method[0] == 'r':
                    scf_obj.xc = self.ct_method[1:]
                scf_obj.small_rho_cutoff = self.rho_cutoff #this prevents pruning. Also slows down code. Can probably remove and use default in pyscf (1e-7)

        self.ct_scf = scf_obj

        # scf settings.
        self.ct_scf.max_cycle = self.cycles
        self.ct_scf.conv_tol = self.conv
        self.ct_scf.conv_tol_grad = self.grad
        self.ct_scf.damp = self.damp
        self.ct_scf.level_shift = self.shift
        self.ct_scf.verbose = self.verbose

        #how to include sigmasmear? Currently not in pyscf

        self.smat = self.ct_scf.get_ovlp()
        self.mo_coeff = [np.zeros_like(self.smat), np.zeros_like(self.smat)]
        self.mo_occ = [np.zeros_like(self.smat[0]), np.zeros_like(self.smat[0])]
        self.mo_energy = self.mo_occ.copy()
        self.fock = self.mo_coeff.copy()
        self.hcore = self.ct_scf.get_hcore()
        self.proj_pot = [[0.0, 0.0] for i in range(len(self.subsystems))]
        self.ct_energy = None

    @time_method("Initialize Densities")
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
            self.get_supersystem_energy()
        s2s = self.sub2sup
        readchk_init = any([subsystem.initguess == 'readchk' or (subsystem.initguess is None and (self.ft_initguess == 'readchk' or self.initguess == 'readchk')) for subsystem in self.subsystems])
        if readchk_init:
            is_chkfile = self.read_chkfile()
        for i in range(len(self.subsystems)):
            dmat = [0.0, 0.0]
            subsystem = self.subsystems[i]
            subsystem.env_scf.grids = self.grids
            if subsystem.initguess is None:
                if self.ft_initguess == 'supmol':
                    dmat[0] = self.dmat[0][np.ix_(s2s[i], s2s[i])]
                    dmat[1] = self.dmat[1][np.ix_(s2s[i], s2s[i])]
                    subsystem.init_density(dmat)

                elif self.ft_initguess == 'readchk':
                    if is_chkfile:
                        if (np.any(subsystem.env_mo_coeff) and np.any(subsystem.env_mo_occ)):
                            dmat[0] = np.dot((subsystem.env_mo_coeff[0] * subsystem.env_mo_occ[0]), subsystem.env_mo_coeff[0].transpose().conjugate())
                            dmat[1] = np.dot((subsystem.env_mo_coeff[1] * subsystem.env_mo_occ[1]), subsystem.env_mo_coeff[1].transpose().conjugate())
                            subsystem.init_density(dmat)
                        elif (np.any(self.mo_coeff) and np.any(self.mo_occ)):
                            sup_dmat[0] = np.dot((self.mo_coeff[0] * self.mo_occ[0]), self.mo_occ[0].transpose().conjugate())
                            sup_dmat[1] = np.dot((self.mo_coeff[1] * self.mo_occ[1]), self.mo_occ[1].transpose().conjugate())
                            dmat[0] = sup_dmat[0][np.ix_(s2s[i], s2s[i])]
                            dmat[1] = sup_dmat[1][np.ix_(s2s[i], s2s[i])]
                            subsystem.init_density(dmat)
                        else:
                            temp_dmat = self.ct_scf.get_init_guess()
                            if temp_dmat.ndim == 2:  #Temp dmat is only one dimensional
                                t_d = [temp_dmat.copy()/2., temp_dmat.copy()/2.]
                                temp_dmat = t_d
                            dmat[0] = temp_dmat[0][np.ix_(s2s[i], s2s[i])]
                            dmat[1] = temp_dmat[1][np.ix_(s2s[i], s2s[i])]
                            subsystem.init_density(dmat)
                    else:
                        temp_dmat = self.ct_scf.get_init_guess()
                        if temp_dmat.ndim == 2:  #Temp dmat is only one dimensional
                            t_d = [temp_dmat.copy()/2., temp_dmat.copy()/2.]
                            temp_dmat = t_d
                        dmat[0] = temp_dmat[0][np.ix_(s2s[i], s2s[i])]
                        dmat[1] = temp_dmat[1][np.ix_(s2s[i], s2s[i])]
                        subsystem.init_density(dmat)
                else:
                    if self.ft_initguess != None:
                        temp_dmat = subsystem.env_scf.get_init_guess(key=self.ft_initguess)
                    else:
                        temp_dmat = subsystem.env_scf.get_init_guess()
                    if temp_dmat.ndim == 2:  #Temp dmat is only one dimensional
                        t_d = [temp_dmat.copy()/2., temp_dmat.copy()/2.]
                        temp_dmat = t_d
                    subsystem.init_density(temp_dmat)

            elif subsystem.initguess == 'supmol': 
                dmat[0] = self.dmat[0][np.ix_(s2s[i], s2s[i])]
                dmat[1] = self.dmat[1][np.ix_(s2s[i], s2s[i])]
                subsystem.init_density(dmat)
            elif subsystem.initguess == 'readchk':
                if is_chkfile:
                    if (np.any(subsystem.env_mo_coeff)  and np.any(subsystem.env_mo_occ)):
                        dmat[0] = np.dot((subsystem.env_mo_coeff[0] * subsystem.env_mo_occ[0]), subsystem.env_mo_coeff[0].transpose().conjugate())
                        dmat[1] = np.dot((subsystem.env_mo_coeff[1] * subsystem.env_mo_occ[1]), subsystem.env_mo_coeff[1].transpose().conjugate())
                        subsystem.init_density(dmat)
                    elif (np.any(self.mo_coeff) and np.any(self.mo_occ)):
                        sup_dmat = [None, None]
                        sup_dmat[0] = np.dot((self.mo_coeff[0] * self.mo_occ[0]), self.mo_coeff[0].transpose().conjugate())
                        sup_dmat[1] = np.dot((self.mo_coeff[1] * self.mo_occ[1]), self.mo_coeff[1].transpose().conjugate())
                        dmat[0] = sup_dmat[0][np.ix_(s2s[i], s2s[i])]
                        dmat[1] = sup_dmat[1][np.ix_(s2s[i], s2s[i])]
                        subsystem.init_density(dmat)
                    else:
                        #temp_dmat = self.ct_scf.get_init_guess()
                        temp_dmat = subsystem.env_scf.get_init_guess()
                        if temp_dmat.ndim == 2:  #Temp dmat is only one dimensional
                            t_d = [temp_dmat.copy()/2., temp_dmat.copy()/2.]
                            temp_dmat = t_d
                        #dmat[0] = temp_dmat[0][np.ix_(s2s[i], s2s[i])]
                        #dmat[1] = temp_dmat[1][np.ix_(s2s[i], s2s[i])]
                        #subsystem.init_density(dmat)
                        subsystem.init_density(temp_dmat)
            else:
                if subsystem.initguess != None:
                    #temp_dmat = self.ct_scf.get_init_guess(key=subsystem.initguess)
                    temp_dmat = subsystem.env_scf.get_init_guess(key=subsystem.initguess)
                else:
                    #temp_dmat = self.ct_scf.get_init_guess()
                    temp_dmat = subsystem.env_scf.get_init_guess()

                if temp_dmat.ndim == 2:  #Temp dmat is only one dimensional
                    t_d = [temp_dmat.copy()/2., temp_dmat.copy()/2.]
                    temp_dmat = t_d
                #dmat[0] = temp_dmat[0][np.ix_(s2s[i], s2s[i])]
                #dmat[1] = temp_dmat[1][np.ix_(s2s[i], s2s[i])]
                #subsystem.init_density(dmat)
                subsystem.init_density(temp_dmat)

            # set core embedding hamiltonian.
            sub_hcore = self.hcore[np.ix_(s2s[i], s2s[i])].copy()
            subsystem.env_hcore = sub_hcore

        #initialize full system density
        if not sup_calc:
            if self.initguess == 'readchk':
                if is_chkfile:
                    if (np.any(self.mo_coeff) and np.any(self.mo_occ)):
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
    

    def concat_mols(self, subsys1=None, subsys2=None):

        # this works but can be done MUCH better.
        mol = gto.Mole()
        mol.basis = {}
        atm = []
        nghost = 0
        mol.charge = 0
        mol.spin = 0
        ghost_atom_list = []
        ghost_basis_dict = {}
        if not (subsys1 is None and subsys2 is None):
            subsys_list = [subsys1, subsys2]
        else:
            subsys_list = self.subsystems
        for i in range(len(subsys_list)):
            subsystem = subsys_list[i]
            mol.charge += subsystem.mol.charge
            mol.spin += subsystem.mol.spin
            for j in range(subsystem.mol.natm):
                if 'ghost' in subsystem.mol.atom_symbol(j).lower():
                    nghost += 1
                    ghost_name = subsystem.mol.atom_symbol(j).split(':')[0] + f':{nghost}'
                    ghost_atom_list.append([ghost_name, subsystem.mol.atom_coord(j)])
                    ghost_basis_dict.update({ghost_name: subsystem.mol.basis[subsystem.mol.atom_symbol(j)]})
                else:
                    if i > 0:
                        atom_name = subsystem.mol.atom_symbol(j) + ':' + str(i)
                    else:
                        atom_name = subsystem.mol.atom_symbol(j)
                    atm.append([atom_name, subsystem.mol.atom_coord(j)])
                    mol.basis.update({atom_name: subsystem.mol.basis[subsystem.mol.atom_symbol(j)]})

        for ghost_atom in ghost_atom_list:
            already_there = False
            for atom in atm:
                if ghost_atom[1][0] == atom[1][0] and ghost_atom[1][1] == atom[1][1] and ghost_atom[1][2] == atom[1][2]:
                    already_there = True

            if not already_there:
                atm.append(ghost_atom)
                mol.basis.update({ghost_atom[0]: ghost_basis_dict[ghost_atom[0]]})

        mol.atom = atm
        mol.verbose = self.verbose
        mol.unit = 'bohr' # atom_coord is always stored in bohr for some reason. Trust me this is right.
        mol.build(dump_input=False)
        return mol

    @time_method("Supersystem Energy")
    def get_supersystem_energy(self):

        if self.ct_energy is None:
            print ("".center(80,'*'))
            print("  Supersystem Calculation  ".center(80))
            print ("".center(80,'*'))
            if self.freeze_and_thaw_conv:  
                nS = self.mol.nao_nr()
                ft_dmat = [np.zeros((nS, nS)), np.zeros((nS, nS))]
                s2s = self.sub2sup
                for i in range(len(self.subsystems)):
                    subsystem = self.subsystems[i]
                    ft_dmat[0][np.ix_(s2s[i], s2s[i])] += subsystem.dmat[0]
                    ft_dmat[1][np.ix_(s2s[i], s2s[i])] += subsystem.dmat[1]
    
                self.ct_scf.scf(dm0=(ft_dmat[0] + ft_dmat[1]))
            else:
                self.ct_scf.scf()
            self.dmat = self.ct_scf.make_rdm1()
            if self.dmat.ndim == 2: #Always store as alpha and beta, even if closed shell. Makes calculations easier.
                t_d = [self.dmat.copy()/2., self.dmat.copy()/2.]
                self.dmat = t_d
                self.mo_coeff = [self.ct_scf.mo_coeff, self.ct_scf.mo_coeff]
                self.mo_occ = [self.ct_scf.mo_occ/2, self.ct_scf.mo_occ/2]
                self.mo_energy = [self.ct_scf.mo_energy, self.ct_scf.mo_energy]
            
            self.save_chkfile()
            self.ct_energy = self.ct_scf.energy_tot()
            print("".center(80,'*'))
        return self.ct_energy

    @time_method("Subsystem Energies")
    def get_emb_subsys_elec_energy(self):

        #Ideally this would be done using the subsystem object, however given how dft energies are calculated this does not seem like a viable option right now.

        #This works. but could be optimized

        #This gets more complex due how DFT exchange correlation energy is calculated.

        nS = self.mol.nao_nr()
        s2s = self.sub2sup
        for i in range(len(self.subsystems)):
            subsystem = self.subsystems[i]
            subsystem.get_env_energy()
            print (f"Uncorrected Energy: {subsystem.env_energy}")
        #self.correct_env_energy()


    @time_method("Environment XC Correction")
    def correct_env_energy(self):
        nS = self.mol.nao_nr()
        s2s = self.sub2sup
        #Correct exc energy
        # get interaction energies.
        for i in range(len(self.subsystems)):
            for j in range(i + 1, len(self.subsystems)):
                # if embedding method is dft, must add the correct embedding energy
                if not 'hf' in self.ct_method[:5]:
                    e_corr_1 = 0.0
                    e_corr_2 = 0.0
                    subsystem_1 = self.subsystems[i]
                    subsystem_2 = self.subsystems[j]
                    dm_subsys = [np.zeros((nS, nS)), np.zeros((nS, nS))]
                    dm_subsys[0][np.ix_(self.sub2sup[i], self.sub2sup[i])] += subsystem_1.dmat[0]
                    dm_subsys[1][np.ix_(self.sub2sup[i], self.sub2sup[i])] += subsystem_1.dmat[1]
                    dm_subsys[0][np.ix_(self.sub2sup[j], self.sub2sup[j])] += subsystem_2.dmat[0]
                    dm_subsys[1][np.ix_(self.sub2sup[j], self.sub2sup[j])] += subsystem_2.dmat[1]

                    dm_subsys_1 = [np.zeros((nS, nS)), np.zeros((nS, nS))]
                    dm_subsys_1[0][np.ix_(self.sub2sup[i], self.sub2sup[i])] += subsystem_1.dmat[0]
                    dm_subsys_1[1][np.ix_(self.sub2sup[i], self.sub2sup[i])] += subsystem_1.dmat[1]

                    dm_subsys_2 = [np.zeros((nS, nS)), np.zeros((nS, nS))]
                    dm_subsys_2[0][np.ix_(self.sub2sup[j], self.sub2sup[j])] += subsystem_2.dmat[0]
                    dm_subsys_2[1][np.ix_(self.sub2sup[j], self.sub2sup[j])] += subsystem_2.dmat[1]

                    #Subtract the incorrect energy
                    #Assume active system is closed shell
                    veff_combined = self.ct_scf.get_veff(dm=(dm_subsys[0] + dm_subsys[1]))
                    vxc_comb = veff_combined
                    vxc_comb = veff_combined - veff_combined.vj
                    if not veff_combined.vk is None:
                        vxc_comb += veff_combined.vk * 0.5

                    veff_active_sub_1 = self.ct_scf.get_veff(dm=(dm_subsys_1[0] + dm_subsys_1[1]))
                    veff_active_sub_2 = self.ct_scf.get_veff(dm=(dm_subsys_2[0] + dm_subsys_2[1]))
                    vxc_sub_1 = veff_active_sub_1 - veff_active_sub_1.vj
                    vxc_sub_2 = veff_active_sub_2 - veff_active_sub_2.vj
                    if not veff_active_sub_1.vk is None:
                        vxc_sub_1 += veff_active_sub_1.vk * 0.5
                    if not veff_active_sub_2.vk is None:
                        vxc_sub_2 += veff_active_sub_2.vk * 0.5

                    vxc_emb_1 = vxc_comb
                    vxc_emb_2 = vxc_comb
                    vxc_emb_1 = vxc_comb - vxc_sub_1
                    vxc_emb_2 = vxc_comb - vxc_sub_2
                    vxc_emb_sub_1 = vxc_emb_1[np.ix_(s2s[i], s2s[i])]
                    vxc_emb_sub_2 = vxc_emb_2[np.ix_(s2s[j], s2s[j])]

                    #Get Exc Last
                    emb_exc_comb_1 = custom_pyscf_methods.exc_rks(self.ct_scf, dm_subsys[0] + dm_subsys[1], dm_subsys_1[0] + dm_subsys_1[1])[1] 
                    emb_exc_comb_2 = custom_pyscf_methods.exc_rks(self.ct_scf, dm_subsys[0] + dm_subsys[1], dm_subsys_2[0] + dm_subsys_2[1])[1] 
                    emb_exc_1 = custom_pyscf_methods.exc_rks(self.ct_scf, dm_subsys_1[0] + dm_subsys_1[1], dm_subsys_1[0] + dm_subsys_1[1])[1] 
                    emb_exc_2 = custom_pyscf_methods.exc_rks(self.ct_scf, dm_subsys_2[0] + dm_subsys_2[1], dm_subsys_2[0] + dm_subsys_2[1])[1] 

                    #vxc_sub_1 = np.einsum('ij,ji', vxc_emb_sub_1, (subsystem_1.dmat[0] + subsystem_1.dmat[1])) #* .5
                    #vxc_sub_2 = np.einsum('ij,ji', vxc_emb_sub_2, (subsystem_2.dmat[0] + subsystem_2.dmat[1])) #* .5

                    subsystem_1.env_energy -= np.einsum('ij,ji', vxc_emb_sub_1, (subsystem_1.dmat[0] + subsystem_1.dmat[1])) #* 0.5
                    subsystem_2.env_energy -= np.einsum('ij,ji', vxc_emb_sub_2, (subsystem_2.dmat[0] + subsystem_2.dmat[1])) #* 0.5
                    subsystem_1.env_energy += (emb_exc_comb_1 - emb_exc_1) * 2.
                    subsystem_2.env_energy += (emb_exc_comb_2 - emb_exc_2) * 2.

    @time_method("Active Energy")
    def get_active_energy(self):
        #This is crude. 
        print ("".center(80,'*'))
        print("  Active Subsystem Calculation  ".center(80))
        print ("".center(80,'*'))
        self.subsystems[0].active_in_env_energy()
        print (f"Uncorrected Energy: {self.subsystems[0].active_energy}")
        #CORRECT ACTIVE SETTINGS.
        #act_elec_e = self.correct_active_energy()
        act_elec_e = 0.0
        self.subsystems[0].active_energy += act_elec_e

        act_e = self.subsystems[0].active_energy
        print(f"Energy: {act_e}")
        print("".center(80,'*'))

    @time_method("Correct Exc Energy")
    def correct_active_energy(self):
        #Assumes all are closed shell systems
        # Do for all subsystems
        nS = self.mol.nao_nr()
        s2s = self.sub2sup
        energy_corr = 0.0
        for j in range(1, len(self.subsystems)):
            # if embedding method is dft, must add the correct embedding energy
            if not 'hf' in self.ct_method[:5]:
                subsystem_1 = self.subsystems[0]
                subsystem_2 = self.subsystems[j]
                dm_subsys = [np.zeros((nS, nS)), np.zeros((nS, nS))]
                dm_subsys[0][np.ix_(self.sub2sup[0], self.sub2sup[0])] += subsystem_1.active_dmat[0]
                dm_subsys[1][np.ix_(self.sub2sup[0], self.sub2sup[0])] += subsystem_1.active_dmat[1]
                dm_subsys[0][np.ix_(self.sub2sup[j], self.sub2sup[j])] += subsystem_2.dmat[0]
                dm_subsys[1][np.ix_(self.sub2sup[j], self.sub2sup[j])] += subsystem_2.dmat[1]

                dm_subsys_1 = [np.zeros((nS, nS)), np.zeros((nS, nS))]
                dm_subsys_1[0][np.ix_(self.sub2sup[0], self.sub2sup[0])] += subsystem_1.active_dmat[0]
                dm_subsys_1[1][np.ix_(self.sub2sup[0], self.sub2sup[0])] += subsystem_1.active_dmat[1]

                #Assume active system is closed shell
                veff_combined = self.ct_scf.get_veff(dm=(dm_subsys[0] + dm_subsys[1]))
                vxc_comb = veff_combined - veff_combined.vj
                if not veff_combined.vk is None:
                    vxc_comb += veff_combined.vk * 0.5

                veff_active_sub = self.ct_scf.get_veff(dm=(dm_subsys_1[0] + dm_subsys_1[1]))
                vxc_sub = veff_active_sub - veff_active_sub.vj
                if not veff_active_sub.vk is None:
                    vxc_sub += veff_active_sub.vk * 0.5

                vxc_emb = vxc_comb - vxc_sub
                vxc_emb_sub = vxc_emb[np.ix_(s2s[0], s2s[0])]

                emb_exc = custom_pyscf_methods.exc_rks(self.ct_scf, dm_subsys[0] + dm_subsys[1], dm_subsys_1[0] + dm_subsys_1[1])[1] 
                emb_exc_a = custom_pyscf_methods.exc_rks(self.ct_scf, dm_subsys_1[0] + dm_subsys_1[1], dm_subsys_1[0] + dm_subsys_1[1])[1] 

                energy_corr -= np.einsum('ij,ji',vxc_emb_sub, (subsystem_1.active_dmat[0] + subsystem_1.active_dmat[1])) #* 0.5
                energy_corr += (emb_exc - emb_exc_a) * 2.
        
                #vxc_emb_sub = vxc_emb[np.ix_(s2s[0], s2s[0])]
                ##THis is incorrect for more than 2 subsys 
                #emb_exc = custom_pyscf_methods.exc_rks(self.ct_scf, dm_subsys[0] + dm_subsys[1], dm_subsys_1[0] + dm_subsys_1[1])[1] 
                #emb_exc_a = custom_pyscf_methods.exc_rks(self.ct_scf, dm_subsys_1[0] + dm_subsys_1[1], dm_subsys_1[0] + dm_subsys_1[1])[1] 
                #
                ##Subtract the Vxc from the fock and recalculate the energies.
                #if 'hf' in subsystem_1.active_method:
               #    emb_pot = (subsystem_1.emb_pot[0] + subsystem_1.emb_pot[1])/2. - vxc_emb_sub
                #    proj_pot = (subsystem_1.proj_pot[0] + subsystem_1.proj_pot[1])/2.
                #    energy_elec = custom_pyscf_methods.rhf_energy_elec(subsystem_1.active_scf, emb_pot, proj_pot)
                #    energy_corr = energy_elec[0] + (emb_exc - emb_exc_a)
                #elif subsystem_1.active_method == 'ccsd' or subsystem_1.active_method == 'rccsd':
                #    new_eris = subsystem_1.active_cc.ao2mo()
                #    emb_pot = (subsystem_1.emb_pot[0] + subsystem_1.emb_pot[1])/2. - vxc_emb_sub
                #    proj_pot = (subsystem_1.proj_pot[0] + subsystem_1.proj_pot[1])/2.
                #    fock = custom_pyscf_methods.rhf_get_fock(subsystem_1.active_scf, emb_pot, proj_pot)
                #    new_eris.fock = functools.reduce(np.dot, (subsystem_1.active_scf.mo_coeff.conj().T, fock, subsystem_1.active_scf.mo_coeff))
                #    energy_elec = custom_pyscf_methods.rhf_energy_elec(subsystem_1.active_scf, emb_pot, proj_pot)[0]
                #    energy_elec += subsystem_1.active_cc.energy(eris=new_eris)  
                #    energy_corr = energy_elec + (emb_exc - emb_exc_a)

                #else:
                #    energy_elec = 0.0
                #    energy_corr = energy_elec + (emb_exc - emb_exc_a)
        
                #Calculate the Exc using the active density and add it to the energies.

        return energy_corr
                

    @time_method("Env. in Env. Energy")
    def env_in_env_energy(self):
        print ("".center(80,'*'))
        print("  Env-in-Env Calculation  ".center(80))
        print ("".center(80,'*'))
        nS = self.mol.nao_nr()
        dm_env = [np.zeros((nS, nS)), np.zeros((nS, nS))]
        for i in range(len(self.subsystems)):
            dm_env[0][np.ix_(self.sub2sup[i], self.sub2sup[i])] += self.subsystems[i].dmat[0]
            dm_env[1][np.ix_(self.sub2sup[i], self.sub2sup[i])] += self.subsystems[i].dmat[1]
        if self.ct_method[0] == 'u' or self.ct_method[:2] == 'ro':
            self.env_energy = self.ct_scf.energy_tot(dm=dm_env)
        else:
            self.env_energy = self.ct_scf.energy_tot(dm=(dm_env[0] + dm_env[1]))

        print(f"  Energy: {self.env_energy}  ".center(80))
        print("".center(80,'*'))
        return self.env_energy


    def update_fock(self, diis=True):

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
            V_a = self.ct_scf.get_veff(mol=self.mol, dm=(dm[0] + dm[1]))
            V_b = V_a

        self.fock[0] += V_a
        self.fock[1] += V_b

        # to use the scf diis methods, must generate individual fock parts and recombine to make full fock matrix, because the projection operator is necessary to use diis correctly.
        #CHANGED TO TEST DIIS METHODS
        if not self.ft_diis is None and diis:
            if self.ct_method[0] == 'u' or self.ct_method[:2] == 'ro':
                self.fock[0] = self.ft_diis[0].update(self.fock[0])
                self.fock[1] = self.ft_diis[0].update(self.fock[1])
            else:
                #f = self.ft_diis[0].update(self.smat, (dm[0] + dm[1]), self.fock[0], self.ct_scf, self.hcore, V_a)
                f = self.ft_diis[0].update(self.fock[0])
                self.fock[0] = f
                self.fock[1] = f
        #if not self.ft_diis is None and diis:
        #    #f = self.ft_diis[0].update(self.smat, (dm[0] + dm[1]), self.fock[0], self.ct_scf, self.hcore, V_a)
        #    f = self.ft_diis[0].update(self.fock[0])
        #    self.fock[0] = f
        #    self.fock[1] = f

    def update_proj_pot(self):
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
                    POp[0] += -1. * ( FDS[0] + FDS[0].transpose() ) 
                    POp[1] += -1. * ( FDS[0] + FDS[0].transpose() )

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
                    POp[0] += -1. * ( FDS[0] + FDS[0].transpose() ) 
                    POp[1] += -1. * ( FDS[1] + FDS[1].transpose() )

            self.proj_pot[i] = POp.copy()

    def read_chkfile(self):
        if os.path.isfile(self.chk_filename):
            with h5py.File(self.chk_filename, 'r') as hf:
                supsys_coeff = hf['supersystem/mo_coeff']
                self.mo_coeff = supsys_coeff[:]
                supsys_occ = hf['supersystem/mo_occ']
                self.mo_occ = supsys_occ[:]
                supsys_energy = hf['supersystem/mo_energy']
                self.mo_energy = supsys_energy[:]

                for i in range(len(self.subsystems)):
                    subsystem = self.subsystems[i]
                    subsys_coeff = hf[f'subsystem:{i}/mo_coeff']
                    subsystem.env_mo_coeff = subsys_coeff[:]
                    subsys_occ = hf[f'subsystem:{i}/mo_occ']
                    subsystem.env_mo_occ = subsys_occ[:]
                    subsys_energy = hf[f'subsystem:{i}/mo_energy']
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
                supsys_coeff[...] = self.mo_coeff
                supsys_occ = hf['supersystem/mo_occ']
                supsys_occ[...] = self.mo_occ
                supsys_energy = hf['supersystem/mo_energy']
                supsys_energy[...] = self.mo_energy
                for i in range(len(self.subsystems)):
                    subsystem = self.subsystems[i]
                    subsys_coeff = hf[f'subsystem:{i}/mo_coeff']
                    subsys_coeff[...] = subsystem.env_mo_coeff
                    subsys_occ = hf[f'subsystem:{i}/mo_occ']
                    subsys_occ[...] = subsystem.env_mo_occ
                    subsys_energy = hf[f'subsystem:{i}/mo_energy']
                    subsys_energy[...] = subsystem.env_mo_energy

        else:
            with h5py.File(self.chk_filename, 'w') as hf:
                hf.create_dataset("embedding_chkfile", data=True)
                sup_mol = hf.create_group('supersystem')
                sup_mol.create_dataset('mo_coeff', data=self.mo_coeff)
                sup_mol.create_dataset('mo_occ', data=self.mo_occ)
                sup_mol.create_dataset('mo_energy', data=self.mo_energy)

                for i in range(len(self.subsystems)):
                    subsystem = self.subsystems[i]
                    sub_sys_data = hf.create_group(f'subsystem:{i}')
                    sub_sys_data.create_dataset('mo_coeff', data=subsystem.env_mo_coeff)
                    sub_sys_data.create_dataset('mo_occ', data=subsystem.env_mo_occ)
                    sub_sys_data.create_dataset('mo_energy', data=subsystem.env_mo_energy)

    @time_method("Freeze and Thaw")
    def freeze_and_thaw(self):
        #Optimization: rather than recalculate vA use the existing fock and subtract out the block that is double counted.

        print("".center(80, '*'))
        print("Freeze-and-Thaw".center(80))
        print("".center(80, '*'))
         
        s2s = self.sub2sup
        ft_err = 1.
        ft_iter = 0 
        last_cycle = False
        while((ft_err > self.ft_conv) and (ft_iter < self.ft_cycles)):
            # cycle over subsystems
            ft_err = 0
            ft_iter += 1
            for i in range(len(self.subsystems)):
                subsystem = self.subsystems[i]
                if not subsystem.freeze:
                    if self.ft_updatefock >= i:
                        self.update_fock(diis=True)

                    subsystem.update_fock()

                    #this will slow down calculation. 
                    if self.analysis:
                        self.get_emb_subsys_elec_energy()
                        sub_old_e = subsystem.get_env_energy()

                    sub_old_dm = subsystem.dmat.copy()

                    self.update_proj_pot() #could use i as input and only get for that sub.
                    FAA = [None, None]
                    FAA[0] = self.fock[0][np.ix_(s2s[i], s2s[i])]
                    FAA[1] = self.fock[1][np.ix_(s2s[i], s2s[i])]
             
                    SAA = self.smat[np.ix_(s2s[i], s2s[i])]
                    #I don't think this changes. Could probably set in the initialize.

                    froz_veff = [None, None]
                    froz_veff[0] = (FAA[0] - subsystem.env_hcore - subsystem.env_V[0])
                    froz_veff[1] = (FAA[1] - subsystem.env_hcore - subsystem.env_V[1])
                    subsystem.update_emb_pot(froz_veff)
                    subsystem.update_proj_pot(self.proj_pot[i])
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
                    if self.analysis:
                        self.get_emb_subsys_elec_energy()
                        sub_new_e = subsystem.get_env_energy()
                        dE = abs(sub_old_e - sub_new_e)

                    # print output to console.
                    if self.analysis:
                        print(f"iter: {ft_iter:>3d}:{i:<2d}  |dE|: {dE:12.6e}   |ddm|: {ddm:12.6e}   |Tr[DP]|: {proj_e:12.6e}")
                    else:
                        print(f"iter: {ft_iter:>3d}:{i:<2d}  |ddm|: {ddm:12.6e}   |Tr[DP]|: {proj_e:12.6e}")

        print("".center(80))
        self.freeze_and_thaw_conv = True
        if(ft_err > self.ft_conv):
            print("".center(80))
            print("Freeze-and-Thaw NOT converged".center(80))
        #Correct for DIIS 
        # cycle over subsystems
        self.update_fock(diis=False)
        for i in range(len(self.subsystems)):
            subsystem = self.subsystems[i]
            if not subsystem.freeze:
                subsystem.update_fock()

                self.update_proj_pot() #could use i as input and only get for that sub.
                FAA = [None, None]
                FAA[0] = self.fock[0][np.ix_(s2s[i], s2s[i])]
                FAA[1] = self.fock[1][np.ix_(s2s[i], s2s[i])]
         
                SAA = self.smat[np.ix_(s2s[i], s2s[i])]
                #I don't think this changes. Could probably set in the initialize.

                froz_veff = [None, None]
                froz_veff[0] = (FAA[0] - subsystem.env_hcore - subsystem.env_V[0])
                froz_veff[1] = (FAA[1] - subsystem.env_hcore - subsystem.env_V[1])
                subsystem.update_emb_pot(froz_veff)
                subsystem.update_proj_pot(self.proj_pot[i])
                subsystem.diagonalize(run_diis=False)
                # save to file. could be done in larger cycles.
                self.save_chkfile()
                self.ft_fermi[i] = subsystem.fermi

        self.get_emb_subsys_elec_energy()
        # print subsystem energies 
        for i in range(len(self.subsystems)):
            subsystem = self.subsystems[i]
            print(f"Subsystem {i} Energy: {subsystem.env_energy:12.8f}")
        print("".center(80))
        print("".center(80, '*'))

        #Assumes closed shell.
        if self.ft_writeorbs:
            print('Writing Subsystem Env Densities'.center(80))
            for i in range(len(self.subsystems)):
                 cubename = os.path.splitext(self.filename)[0] + '_' + str(i+1) + '.cube'
                 subsystem = self.subsystems[i]
                 dmat = subsystem.dmat[0] + subsystem.dmat[1]
                 cubegen.density(subsystem.mol, cubename, dmat) 
