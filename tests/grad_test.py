# A test to be tried later.


import unittest
import os
import shutil
import re

from copy import copy

from qsome import cluster_subsystem, cluster_supersystem
from pyscf import gto, lib, scf, dft
import numpy as np

#Open Shell
#propyl radical
mol = gto.Mole()
mol.atom = '''
C  0.5000  0.2500  2.8660
'''
mol.basis = '3-21g'
mol.spin = 2
#mol.verbose = 5
mol.charge = -2
mol.build()
#env_method = 'b3lyp'
env_method = 'hf'
hl_method = 'hf'
subsys = cluster_subsystem.ClusterHLSubSystem(mol, env_method, hl_method, hl_unrestricted=True, unrestricted=True)

mol2 = gto.Mole()
#mol2.verbose = 4
mol2.atom = '''
C  0.0000  0.7500  3.7320
C  0.0000  0.7500  2.0000
H  0.0000  0.2131  4.0420
H  0.0000  1.0600  4.2690
H  0.0000  1.2869  3.4220
H  0.0000  1.2869  2.3100
H  0.0000  1.0600  1.4631
H  0.0000  0.2131  1.6900
'''
mol2.basis = '3-21g'
mol2.charge = 2
mol2.build()
subsys2 = cluster_subsystem.ClusterEnvSubSystem(mol2, env_method, unrestricted=True)
mol_total = gto.Mole()
mol_total.atom = '''
C  0.5000  0.2500  2.8660
C  0.0000  0.7500  3.7320
C  0.0000  0.7500  2.0000
H  0.0000  0.2131  4.0420
H  0.0000  1.0600  4.2690
H  0.0000  1.2869  3.4220
H  0.0000  1.2869  2.3100
H  0.0000  1.0600  1.4631
H  0.0000  0.2131  1.6900
'''
mol_total.basis = '3-21g'
mol_total.spin = 2
#mol_total.verbose = 5
mol_total.build()
super_hf = scf.UHF(mol_total)
#super_dft = dft.UKS(mol_total)
#super_dft.xc = 'b3lyp'
#super_dft.kernel()
#dm = super_dft.make_rdm1()
#super_dft_grad = super_dft.nuc_grad_method()
#super_dft_grad.grid_response = True
#super_dft_grad.kernel()
#print (x)
#ni = super_dft._numint
##Text nr_fxc method
#vmat = ni.nr_fxc(mol_total, super_dft.grids, 'lda', dm, dm, spin=2, verbose=9)
#
#print (vmat[0] * 3.) #Why is this off by a factor of 3?
#vhf = super_dft.get_veff()
##print (vhf.vj.shape)
#print (vhf[0] - vhf.vj)
#print (vhf[1] - vhf.vj)
#nao,nmo = super_dft.mo_coeff[0].shape
#rho0, vxc, fxc = ni.cache_xc_kernel(mol_total, super_dft.grids, super_dft.xc, super_dft.mo_coeff, super_dft.mo_occ, mol_total.spin)
#print (len(rho0))
#print (len(vxc))
#print (len(fxc))

#Test the dft kernel.
#super_dft.grids.level = 3
#super_hf = scf.UHF(mol_total)
#supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'b3lyp', super_dft, diis_num=2, conv_tol=1e-8)
supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'hf', super_hf, diis_num=2, conv_tol=1e-8)
supersystem.init_density()
supsystem_e = supersystem.get_supersystem_energy()
supersystem.freeze_and_thaw()
# confirm projection and embed potential.
sba = gto.intor_cross('int1e_ovlp', subsys2.mol, subsys.mol)
hcore_emb = gto.intor_cross('int1e_kin', subsys.mol, subsys2.mol)
hcore_emb += gto.intor_cross('int1e_nuc', subsys.mol, subsys2.mol)
print (hcore_emb.shape)

ao_b = supersystem.mol.intor('int2e', shls_slice=(0,subsys.mol.nbas, subsys.mol.nbas,subsys.mol.nbas+subsys2.mol.nbas,
                                             subsys.mol.nbas, subsys.mol.nbas+subsys2.mol.nbas, subsys.mol.nbas, subsys.mol.nbas+subsys2.mol.nbas))

j_b = np.einsum('ijkl,kl->ij', ao_b, subsys2.env_dmat[0] + subsys2.env_dmat[1])
k_b = np.einsum('ijkl,sjk->sil', ao_b, subsys2.env_dmat)
ao_a = supersystem.mol.intor('int2e', shls_slice=(0,subsys.mol.nbas, subsys.mol.nbas,subsys.mol.nbas+subsys2.mol.nbas,
                                                  0, subsys.mol.nbas, 0, subsys.mol.nbas))
j_a = np.einsum('ijkl,kl->ij', ao_a, subsys.env_dmat[0] + subsys.env_dmat[1])
k_a = np.einsum('ijkl,sil->skj', ao_a, subsys.env_dmat)
print (j_b.shape)
print (k_b.shape)
print (j_a.shape)
print (k_a.shape)
subsys_dmat0 = np.zeros_like(subsys.env_dmat[0])
for m in range(subsys.env_mo_coeff[0].shape[0]):
    for n in range(subsys.env_mo_coeff[0].shape[1]):
        for i in range(subsys.mol.nelec[0]):
            subsys_dmat0[m,n] += subsys.env_mo_coeff[0][m,i] * subsys.env_mo_coeff[0][n,i]
print (subsys_dmat0)
print (subsys.env_dmat[0])
proj_test = [None, None]
f_ab = hcore_emb + j_b + j_a -k_b[0] - k_a[0]
s2s = supersystem.sub2sup
print (supersystem.fock[0][np.ix_(s2s[0],s2s[1])])
print (np.max(f_ab - supersystem.fock[0][np.ix_(s2s[0],s2s[1])]))
proj_test[0] = np.linalg.multi_dot([hcore_emb + j_b + j_a - k_b[0] - k_a[0], subsys2.env_dmat[0], sba])
print (proj_test[0] + proj_test[0].T)
print (subsys.proj_pot[0])



#supersystem_grad = supersystem.get_emb_nuc_grad()
#subsystem_grad = subsys.get_env_nuc_grad()
#subsys.active_in_env_energy()
#subsys.get_active_nuc_grad()
#total_energy = supsystem_e - supersystem.subsystems[0].env_energy + supersystem.subsystems[0].active_energy
#print (total_energy)
#
#
