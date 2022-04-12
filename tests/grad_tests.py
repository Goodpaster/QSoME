# Tests for the analytical nuclear gradients
# Daniel Graham

import unittest
import os
import shutil
import re
import copy
import numpy as np

from functools import reduce
from qsome import cluster_supersystem, cluster_subsystem, helpers
from pyscf import gto

class TestEnvSubSystemGradients(unittest.TestCase):

    def setUp(self):

        cs_mol = gto.M()
        cs_mol.atom = '''
        N    1.7030   29.2921   -0.4884
        '''
        cs_mol.basis = 'cc-pVDZ'
        cs_mol.charge = -3
        cs_mol.build()
        self.cs_small_mol1 = cs_mol

        cs_mol = gto.M()
        cs_mol.atom = '''
        F    0.1341   29.9216    0.1245
        F    1.1937   28.0518    0.1245
        F    2.2823   29.9034    0.1243
        '''
        cs_mol.basis = 'cc-pVDZ'
        cs_mol.charge = 3
        cs_mol.build()
        self.cs_small_mol2 = cs_mol

        cs_mol = gto.M()
        cs_mol.atom = '''
        H    1.1979    1.7205    1.1857
        C    0.9565    0.6866    0.9272
        '''
        cs_mol.basis = 'cc-pVDZ'
        cs_mol.spin = -3
        cs_mol.build()
        self.cs_sub0_twoatm_mol1 = cs_mol

        cs_mol = gto.M()
        cs_mol.atom = '''
        F    2.0412    0.0602    0.4450
        F   -0.0071    0.6514   -0.0062
        F    0.5258    0.0225    2.0110
        '''
        cs_mol.basis = 'cc-pVDZ'
        cs_mol.charge = 3
        cs_mol.build()
        self.cs_sub1_twoatm_mol1 = cs_mol

        os_mol = gto.M()
        os_mol.atom = '''
        C    1.2026    0.7046    0.0000
        '''
        os_mol.basis = 'cc-pVDZ'
        os_mol.spin = 2
        os_mol.charge = -2
        os_mol.build()
        self.os_sub0_small_mol1 = os_mol

        os_mol = gto.M()
        os_mol.atom = '''
        F    0.8919    1.9804    0.0000 
        F    0.0403    0.0932    0.0000
        '''
        os_mol.basis = 'cc-pVDZ'
        os_mol.charge = 2
        os_mol.build()
        self.os_sub1_small_mol1 = os_mol

        os_mol = gto.M()
        os_mol.atom = '''
        H    1.8306    0.3589    0.8863
        C    1.2048    0.6943    0.0548
        '''
        os_mol.basis = 'cc-pVDZ'
        os_mol.spin = 1
        os_mol.charge = -2
        os_mol.build()
        self.os_sub0_twoatm_mol1 = os_mol

        os_mol = gto.M()
        os_mol.atom = '''
        F    0.0261    0.0744   -0.0159
        F    1.0682    2.0190   -0.0159
        '''
        os_mol.basis = 'cc-pVDZ'
        os_mol.charge = 2
        os_mol.build()
        self.os_sub1_twoatm_mol1 = os_mol

    #def test_rhf_proj_grad(self):
    #    env_method = 'hf'
    #    hl_method = 'ccsd'
    #    subsys1 = cluster_subsystem.ClusterHLSubSystem(self.cs_small_mol1, env_method, hl_method)
    #    env_subsys1 = cluster_subsystem.ClusterEnvSubSystem(self.cs_small_mol2, env_method)
    #    sup_mol11 = helpers.concat_mols([self.cs_small_mol1, self.cs_small_mol2])
    #    fs_scf_obj_1 = helpers.gen_scf_obj(sup_mol11, env_method)
    #    supersystem = cluster_supersystem.ClusterSuperSystem([subsys1, env_subsys1], env_method, fs_scf_obj_1)
    #    supersystem.init_density()
    #    supersystem.freeze_and_thaw()

    #    #Get numerical terms.
    #    x_dir_diff = 0.000001
    #    coord_0 = self.cs_small_mol1.atom_coords()
    #    coord_0[0][0] = coord_0[0][0] - x_dir_diff
    #    mol0 = self.cs_small_mol1.set_geom_(coord_0, 'B', inplace=False)
    #    subsys0 = cluster_subsystem.ClusterHLSubSystem(mol0, env_method, hl_method)
    #    env_mol0 = copy.copy(self.cs_small_mol2)
    #    env_subsys0 = cluster_subsystem.ClusterEnvSubSystem(env_mol0, env_method)
    #    sup_mol00 = helpers.concat_mols([mol0, env_mol0])
    #    fs_scf_obj0 = helpers.gen_scf_obj(sup_mol00, env_method)
    #    supersystem0 = cluster_supersystem.ClusterSuperSystem([subsys0, env_subsys0], env_method, fs_scf_obj0)
    #    supersystem0.init_density()
    #    supersystem0.freeze_and_thaw()

    #    coord_2 = self.cs_small_mol1.atom_coords()
    #    coord_2[0][0] = coord_2[0][0] + x_dir_diff
    #    mol2 = self.cs_small_mol1.set_geom_(coord_2, 'B', inplace=False)
    #    subsys2 = cluster_subsystem.ClusterHLSubSystem(mol2, env_method, hl_method)
    #    env_mol2 = copy.copy(self.cs_small_mol2)
    #    env_subsys2 = cluster_subsystem.ClusterEnvSubSystem(env_mol2, env_method)
    #    sup_mol22 = helpers.concat_mols([mol2, env_mol2])
    #    fs_scf_obj2 = helpers.gen_scf_obj(sup_mol22, env_method)
    #    supersystem2 = cluster_supersystem.ClusterSuperSystem([subsys2, env_subsys2], env_method, fs_scf_obj2)
    #    supersystem2.init_density()
    #    supersystem2.freeze_and_thaw()

    #    num_sub1_den_grad = np.zeros((2,3,subsys2.env_dmat[0].shape[0], subsys2.env_dmat[0].shape[1]))
    #    num_sub2_den_grad = np.zeros((2,3,env_subsys2.env_dmat[0].shape[0], env_subsys2.env_dmat[0].shape[1]))
    #    num_sub1_den_grad[0,0] = (np.array(subsys2.env_dmat)[0] - np.array(subsys0.env_dmat)[0])/(x_dir_diff*2.)
    #    num_sub1_den_grad[1,0] = (np.array(subsys2.env_dmat)[1] - np.array(subsys0.env_dmat)[1])/(x_dir_diff*2.)
    #    num_sub2_den_grad[0,0] = (np.array(env_subsys2.env_dmat)[0] - np.array(env_subsys0.env_dmat)[0])/(x_dir_diff*2.)
    #    num_sub2_den_grad[1,0] = (np.array(env_subsys2.env_dmat)[1] - np.array(env_subsys0.env_dmat)[1])/(x_dir_diff*2.)

    #    sub0s2s = supersystem0.sub2sup
    #    sub2s2s = supersystem0.sub2sup
    #    num_sup_smat_grad = np.zeros((3,supersystem0.smat.shape[0], supersystem0.smat.shape[1]))
    #    num_sup_smat_grad[0][np.ix_(sub2s2s[1],sub2s2s[0])] += (supersystem2.smat[np.ix_(sub2s2s[1],sub2s2s[0])] - supersystem0.smat[np.ix_(sub0s2s[1],sub0s2s[0])])/(x_dir_diff*2.)

    #    num_sub1_proj_grad = np.zeros((2,3,subsys2.env_dmat[0].shape[0], subsys2.env_dmat[0].shape[1]))
    #    num_sub1_proj_grad[0,0] = (np.array(subsys2.proj_pot)[0] - np.array(subsys0.proj_pot)[0])/(x_dir_diff*2.)
    #    num_sub1_proj_grad[1,0] = (np.array(subsys2.proj_pot)[1] - np.array(subsys0.proj_pot)[1])/(x_dir_diff*2.)

    #    num_sub1_proj_hcore_grad = np.zeros((2,3,subsys2.env_dmat[0].shape[0], subsys2.env_dmat[0].shape[1]))
    #    sub0_hcore_ab = supersystem0.hcore[np.ix_(sub0s2s[0], sub0s2s[1])]
    #    sub0_smat_ba = supersystem0.smat[np.ix_(sub0s2s[1], sub0s2s[0])]
    #    sub2_hcore_ab = supersystem2.hcore[np.ix_(sub2s2s[0], sub2s2s[1])]
    #    sub2_smat_ba = supersystem2.smat[np.ix_(sub2s2s[1], sub2s2s[0])]
    #    sub0_hcore_den_smat = [None, None]
    #    sub2_hcore_den_smat = [None, None]
    #    sub0_hcore_den_smat[0] = np.dot(sub0_hcore_ab, np.dot(env_subsys0.env_dmat[0], sub0_smat_ba))
    #    sub0_hcore_den_smat[1] = np.dot(sub0_hcore_ab, np.dot(env_subsys0.env_dmat[1], sub0_smat_ba))
    #    sub2_hcore_den_smat[0] = np.dot(sub2_hcore_ab, np.dot(env_subsys2.env_dmat[0], sub2_smat_ba))
    #    sub2_hcore_den_smat[1] = np.dot(sub2_hcore_ab, np.dot(env_subsys2.env_dmat[1], sub2_smat_ba))
    #    sub0_hcore_proj_op = [None, None]
    #    sub0_hcore_proj_op[0] = -1 * (sub0_hcore_den_smat[0] + sub0_hcore_den_smat[0].transpose())
    #    sub0_hcore_proj_op[1] = -1 * (sub0_hcore_den_smat[1] + sub0_hcore_den_smat[1].transpose())
    #    sub2_hcore_proj_op = [None, None]
    #    sub2_hcore_proj_op[0] = -1 * (sub2_hcore_den_smat[0] + sub2_hcore_den_smat[0].transpose())
    #    sub2_hcore_proj_op[1] = -1 * (sub2_hcore_den_smat[1] + sub2_hcore_den_smat[1].transpose())

    #    num_sub1_proj_hcore_grad[:,0] = (np.array(sub2_hcore_proj_op) - np.array(sub0_hcore_proj_op))/(x_dir_diff*2.)

    #    #Test occ part.
    #    sub0_fock_ab = [None, None]
    #    sub0_fock_ab[0] = supersystem0.fock[0][np.ix_(sub0s2s[0], sub0s2s[1])]
    #    sub0_fock_ab[1] = supersystem0.fock[1][np.ix_(sub0s2s[0], sub0s2s[1])]
    #    sub0_smat_ba = supersystem0.smat[np.ix_(sub0s2s[1], sub0s2s[0])]
    #    sub0_fock_ab_mj = np.zeros_like(sub0_fock_ab)
    #    sub0_fock_ab_mj[0] = np.dot(sub0_fock_ab[0], env_subsys0.env_mo_coeff[0])
    #    sub0_fock_ab_mj[1] = np.dot(sub0_fock_ab[1], env_subsys0.env_mo_coeff[1])
    #    sub0_smat_ba_jn = np.zeros_like([sub0_smat_ba, sub0_smat_ba])
    #    sub0_smat_ba_jn[0] = np.dot(env_subsys0.env_mo_coeff[0].T, sub0_smat_ba)
    #    sub0_smat_ba_jn[1] = np.dot(env_subsys0.env_mo_coeff[1].T, sub0_smat_ba)
    #    occa,occb = env_subsys0.mol.nelec
    #    sub0_fock_ab_mj_occ = [None, None]
    #    sub0_fock_ab_mj_occ[0] = sub0_fock_ab_mj[0,:occa,:]
    #    sub0_fock_ab_mj_occ[1] = sub0_fock_ab_mj[1,:occb,:]
    #    sub0_smat_ba_jn_occ = [None, None]
    #    sub0_smat_ba_jn_occ[0] = sub0_smat_ba_jn[0,:,:occa]
    #    sub0_smat_ba_jn_occ[1] = sub0_smat_ba_jn[1,:,:occb]
    #    sub0_proj_occ = [None, None]
    #    sub0_proj_occ[0] = np.dot(sub0_fock_ab_mj_occ[0], sub0_smat_ba_jn_occ[0])
    #    sub0_proj_occ[1] = np.dot(sub0_fock_ab_mj_occ[1], sub0_smat_ba_jn_occ[1])
    #    sub0_proj_occ[0] += sub0_proj_occ[0].T
    #    sub0_proj_occ[1] += sub0_proj_occ[1].T

    #    sub2_fock_ab = [None, None]
    #    sub2_fock_ab[0] = supersystem2.fock[0][np.ix_(sub2s2s[0], sub2s2s[1])]
    #    sub2_fock_ab[1] = supersystem2.fock[1][np.ix_(sub2s2s[0], sub2s2s[1])]
    #    sub2_smat_ba = supersystem2.smat[np.ix_(sub2s2s[1], sub2s2s[0])]
    #    sub2_fock_ab_mj = np.zeros_like(sub2_fock_ab)
    #    sub2_fock_ab_mj[0] = np.dot(sub2_fock_ab[0], env_subsys2.env_mo_coeff[0])
    #    sub2_fock_ab_mj[1] = np.dot(sub2_fock_ab[1], env_subsys2.env_mo_coeff[1])
    #    sub2_smat_ba_jn = np.zeros_like([sub2_smat_ba, sub2_smat_ba])
    #    sub2_smat_ba_jn[0] = np.dot(env_subsys2.env_mo_coeff[0].T, sub2_smat_ba)
    #    sub2_smat_ba_jn[1] = np.dot(env_subsys2.env_mo_coeff[1].T, sub2_smat_ba)
    #    occa,occb = env_subsys2.mol.nelec
    #    sub2_fock_ab_mj_occ = [None, None]
    #    sub2_fock_ab_mj_occ[0] = sub2_fock_ab_mj[0,:occa,:]
    #    sub2_fock_ab_mj_occ[1] = sub2_fock_ab_mj[1,:occb,:]
    #    sub2_smat_ba_jn_occ = [None, None]
    #    sub2_smat_ba_jn_occ[0] = sub2_smat_ba_jn[0,:,:occa]
    #    sub2_smat_ba_jn_occ[1] = sub2_smat_ba_jn[1,:,:occb]
    #    sub2_proj_occ = [None, None]
    #    sub2_proj_occ[0] = np.dot(sub2_fock_ab_mj_occ[0], sub2_smat_ba_jn_occ[0])
    #    sub2_proj_occ[1] = np.dot(sub2_fock_ab_mj_occ[1], sub2_smat_ba_jn_occ[1])
    #    sub2_proj_occ[0] += sub2_proj_occ[0].T
    #    sub2_proj_occ[1] += sub2_proj_occ[1].T

    #    #D TEST
    #    td = np.dot(env_subsys2.env_mo_coeff[0][:,:occa],env_subsys2.env_mo_coeff[0][:,:occa].T)

    #    occ_proj_grad = (sub2_proj_occ[0] - sub0_proj_occ[0])/(x_dir_diff*2.)
    #    print (occ_proj_grad)
    #    occ_proj_grad = (sub2_proj_occ[1] - sub0_proj_occ[1])/(x_dir_diff*2.)
    #    print (occ_proj_grad)

    #    print ('d + d.T term')
    #    print (np.max(num_sub2_den_grad-num_sub2_den_grad.transpose(0,1,3,2)))
    #    ddt = num_sub2_den_grad + num_sub2_den_grad.transpose(0,1,3,2)
    #    print (ddt[:,0])
    #    print ("COMPARE")
    #    print (num_sub1_proj_hcore_grad[:,0])

    #    ana_sub1_proj_hcore_grad = supersystem.get_sub_proj_grad([num_sub1_den_grad,num_sub2_den_grad])
    #    print (np.max(num_sub1_proj_hcore_grad[:,0] - ana_sub1_proj_hcore_grad[:,0]))
        

    def test_rhf_den_grad(self):
        env_method = 'hf'
        hl_method = 'ccsd'
        subsys1 = cluster_subsystem.ClusterHLSubSystem(self.cs_small_mol1, env_method, hl_method)
        env_subsys1 = cluster_subsystem.ClusterEnvSubSystem(self.cs_small_mol2, env_method)
        sup_mol11 = helpers.concat_mols([self.cs_small_mol1, self.cs_small_mol2])
        fs_scf_obj_1 = helpers.gen_scf_obj(sup_mol11, env_method)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys1, env_subsys1], env_method, fs_scf_obj_1)
        supersystem.init_density()
        supersystem.freeze_and_thaw()

        #Get numerical terms.
        x_dir_diff = 0.000001
        coord_0 = self.cs_small_mol1.atom_coords()
        coord_0[0][0] = coord_0[0][0] - x_dir_diff
        mol0 = self.cs_small_mol1.set_geom_(coord_0, 'B', inplace=False)
        subsys0 = cluster_subsystem.ClusterHLSubSystem(mol0, env_method, hl_method)
        env_mol0 = copy.copy(self.cs_small_mol2)
        env_subsys0 = cluster_subsystem.ClusterEnvSubSystem(env_mol0, env_method)
        sup_mol00 = helpers.concat_mols([mol0, env_mol0])
        fs_scf_obj0 = helpers.gen_scf_obj(sup_mol00, env_method)
        supersystem0 = cluster_supersystem.ClusterSuperSystem([subsys0, env_subsys0], env_method, fs_scf_obj0)
        supersystem0.init_density()
        supersystem0.freeze_and_thaw()

        coord_2 = self.cs_small_mol1.atom_coords()
        coord_2[0][0] = coord_2[0][0] + x_dir_diff
        mol2 = self.cs_small_mol1.set_geom_(coord_2, 'B', inplace=False)
        subsys2 = cluster_subsystem.ClusterHLSubSystem(mol2, env_method, hl_method)
        env_mol2 = copy.copy(self.cs_small_mol2)
        env_subsys2 = cluster_subsystem.ClusterEnvSubSystem(env_mol2, env_method)
        sup_mol22 = helpers.concat_mols([mol2, env_mol2])
        fs_scf_obj2 = helpers.gen_scf_obj(sup_mol22, env_method)
        supersystem2 = cluster_supersystem.ClusterSuperSystem([subsys2, env_subsys2], env_method, fs_scf_obj2)
        supersystem2.init_density()
        supersystem2.freeze_and_thaw()

        #Num energy grad terms
        num_sub1_energy_grad = (subsys2.env_energy - subsys0.env_energy)/(x_dir_diff*2.)
        num_sub2_energy_grad = (env_subsys2.env_energy - env_subsys0.env_energy)/(x_dir_diff*2.)
        num_sup_energy_grad = (supersystem2.fs_energy - supersystem0.fs_energy)/(x_dir_diff*2.)

        #Num density grad terms
        num_sub1_den_grad = np.zeros((2,3,subsys2.env_dmat[0].shape[0], subsys2.env_dmat[0].shape[1]))
        num_sub2_den_grad = np.zeros((2,3,env_subsys2.env_dmat[0].shape[0], env_subsys2.env_dmat[0].shape[1]))
        num_sub1_den_grad[0,0] = (np.array(subsys2.env_dmat)[0] - np.array(subsys0.env_dmat)[0])/(x_dir_diff*2.)
        num_sub1_den_grad[1,0] = (np.array(subsys2.env_dmat)[1] - np.array(subsys0.env_dmat)[1])/(x_dir_diff*2.)
        num_sub2_den_grad[0,0] = (np.array(env_subsys2.env_dmat)[0] - np.array(env_subsys0.env_dmat)[0])/(x_dir_diff*2.)
        num_sub2_den_grad[1,0] = (np.array(env_subsys2.env_dmat)[1] - np.array(env_subsys0.env_dmat)[1])/(x_dir_diff*2.)
        num_sup_den_grad = (np.array(supersystem2.get_emb_dmat()) - np.array(supersystem0.get_emb_dmat()))/(x_dir_diff*2.)

        #Num mo_coeff terms
        num_sub1_mo_coeff_grad = np.zeros((2,3,subsys2.env_mo_coeff[0].shape[0], subsys2.env_mo_coeff[0].shape[1]))
        num_sub1_mo_coeff_grad[0,0] = (np.array(subsys2.env_mo_coeff)[0] - np.array(subsys0.env_mo_coeff)[0])/(x_dir_diff*2.)
        num_sub1_mo_coeff_grad[1,0] = (np.array(subsys2.env_mo_coeff)[1] - np.array(subsys0.env_mo_coeff)[1])/(x_dir_diff*2.)

        num_sub1_ovlp_grad = (subsys2.env_scf.get_ovlp() - subsys0.env_scf.get_ovlp())/(x_dir_diff*2.)

        #test for occ block terms.
        s1a = -subsys1.mol.intor('int1e_ipovlp')
        print ('here')
        print (s1a[0])
        print (s1a.transpose(0,2,1)[0])
        print (s1a[0] + s1a.transpose(0,2,1)[0])
        print (num_sub1_ovlp_grad)
        print (np.max(num_sub1_ovlp_grad - (s1a[0] + s1a[0].transpose())))
        mo_occ = subsys1.env_mo_occ[0]
        mocc = subsys1.env_mo_coeff[0][:,mo_occ>0]
        def _ao2mo(mat):
            return np.asarray([reduce(np.dot, (subsys1.env_mo_coeff[0].T, x, mocc)) for x in mat])

        s1vo = _ao2mo(s1a)

        ovlp_grad = s1a + s1a.transpose(0,2,1)
        num_sub1_ua = np.dot(np.linalg.inv(subsys1.env_mo_coeff[0]), num_sub1_mo_coeff_grad[0,0])
        num_sub1_ub = np.dot(np.linalg.inv(subsys1.env_mo_coeff[1]), num_sub1_mo_coeff_grad[1,0])
        num_smat_grad_mo = np.dot(subsys1.env_mo_coeff[0].T, np.dot(ovlp_grad[0], subsys1.env_mo_coeff[0]))

        inv_coeff = np.linalg.inv(subsys1.env_mo_coeff[0])
        p1 = np.dot(inv_coeff,np.dot(num_sub1_den_grad[0,0], inv_coeff.T))

        #p1 = np.dot(num_sub1_ua[:,mo_occ>0], num_sub1_ua[:,mo_occ>0].T)
        occ = p1[:subsys1.mol.nelec[0],:subsys1.mol.nelec[0]]
        occ_vir = p1[subsys1.mol.nelec[0]:,:subsys1.mol.nelec[0]]
        vir = p1[subsys1.mol.nelec[0]:,subsys1.mol.nelec[0]:]
        #occ_ua = num_sub1_ua[:subsys1.mol.nelec[0],:subsys1.mol.nelec[0]]
        #print (occ)
        #print (vir)
        #print (occ_vir)
        #print (num_sub1_ua[subsys1.mol.nelec[0]:,:subsys1.mol.nelec[0]])
        #print (occ_ua + occ_ua.T)
        #print (x)

        nmoa = subsys1.env_mo_occ[0].size
        occidxa = mo_occ > 0
        nocca = np.count_nonzero(occidxa)
        s1_a = s1vo.reshape(-1,nmoa,nocca)
        s1_a = -s1_a[:,occidxa] * 0.5

        #print (num_sub1_mo_den_grad[0][subsys1.mol.nelec[0]:,subsys1.mol.nelec[0]:])
        supersystem.get_sub_den_grad()

    #def test_rhf_grad(self):
    #    env_method = 'hf'
    #    hl_method = 'ccsd'
    #    subsys1 = cluster_subsystem.ClusterHLSubSystem(self.cs_small_mol1, env_method, hl_method)
    #    env_subsys1 = cluster_subsystem.ClusterEnvSubSystem(self.cs_small_mol2, env_method)
    #    sup_mol11 = helpers.concat_mols([self.cs_small_mol1, self.cs_small_mol2])
    #    fs_scf_obj_1 = helpers.gen_scf_obj(sup_mol11, env_method)
    #    supersystem = cluster_supersystem.ClusterSuperSystem([subsys1, env_subsys1], env_method, fs_scf_obj_1)
    #    supersystem.init_density()
    #    supersystem.freeze_and_thaw()

    #    #Get numerical terms.
    #    x_dir_diff = 0.000001
    #    coord_0 = self.cs_small_mol1.atom_coords()
    #    coord_0[0][0] = coord_0[0][0] - x_dir_diff
    #    mol0 = self.cs_small_mol1.set_geom_(coord_0, 'B', inplace=False)
    #    subsys0 = cluster_subsystem.ClusterHLSubSystem(mol0, env_method, hl_method)
    #    env_mol0 = copy.copy(self.cs_small_mol2)
    #    env_subsys0 = cluster_subsystem.ClusterEnvSubSystem(env_mol0, env_method)
    #    sup_mol00 = helpers.concat_mols([mol0, env_mol0])
    #    fs_scf_obj0 = helpers.gen_scf_obj(sup_mol00, env_method)
    #    supersystem0 = cluster_supersystem.ClusterSuperSystem([subsys0, env_subsys0], env_method, fs_scf_obj0)
    #    supersystem0.init_density()
    #    supersystem0.freeze_and_thaw()

    #    coord_2 = self.cs_small_mol1.atom_coords()
    #    coord_2[0][0] = coord_2[0][0] + x_dir_diff
    #    mol2 = self.cs_small_mol1.set_geom_(coord_2, 'B', inplace=False)
    #    subsys2 = cluster_subsystem.ClusterHLSubSystem(mol2, env_method, hl_method)
    #    env_mol2 = copy.copy(self.cs_small_mol2)
    #    env_subsys2 = cluster_subsystem.ClusterEnvSubSystem(env_mol2, env_method)
    #    sup_mol22 = helpers.concat_mols([mol2, env_mol2])
    #    fs_scf_obj2 = helpers.gen_scf_obj(sup_mol22, env_method)
    #    supersystem2 = cluster_supersystem.ClusterSuperSystem([subsys2, env_subsys2], env_method, fs_scf_obj2)
    #    supersystem2.init_density()
    #    supersystem2.freeze_and_thaw()

    #    #Num energy grad terms
    #    num_sub1_energy_grad = (subsys2.env_energy - subsys0.env_energy)/(x_dir_diff*2.)
    #    num_sub2_energy_grad = (env_subsys2.env_energy - env_subsys0.env_energy)/(x_dir_diff*2.)
    #    num_sup_energy_grad = (supersystem2.fs_energy - supersystem0.fs_energy)/(x_dir_diff*2.)

    #    #Num density grad terms
    #    num_sub1_den_grad = np.zeros((2,3,subsys2.env_dmat[0].shape[0], subsys2.env_dmat[0].shape[1]))
    #    num_sub2_den_grad = np.zeros((2,3,env_subsys2.env_dmat[0].shape[0], env_subsys2.env_dmat[0].shape[1]))
    #    num_sub1_den_grad[0,0] = (np.array(subsys2.env_dmat)[0] - np.array(subsys0.env_dmat)[0])/(x_dir_diff*2.)
    #    num_sub1_den_grad[1,0] = (np.array(subsys2.env_dmat)[1] - np.array(subsys0.env_dmat)[1])/(x_dir_diff*2.)
    #    num_sub2_den_grad[0,0] = (np.array(env_subsys2.env_dmat)[0] - np.array(env_subsys0.env_dmat)[0])/(x_dir_diff*2.)
    #    num_sub2_den_grad[1,0] = (np.array(env_subsys2.env_dmat)[1] - np.array(env_subsys0.env_dmat)[1])/(x_dir_diff*2.)
    #    num_sup_den_grad = (np.array(supersystem2.get_emb_dmat()) - np.array(supersystem0.get_emb_dmat()))/(x_dir_diff*2.)
    #    #Compare terms.

    #    ana_sub_nuc_grad = supersystem.get_emb_nuc_grad(den_grad=(num_sub1_den_grad,num_sub2_den_grad))

    #    print (num_sub1_energy_grad)
    #    print (ana_sub_nuc_grad[0][0])

    #    #larger system.

    def test_uhf_grad(self):
        env_method = 'hf'
        hl_method = 'ccsd'
    def test_rohf_grad(self):
        env_method = 'hf'
        hl_method = 'ccsd'
    def test_rks_grad(self):
        pass
    def test_uks_grad(self):
        pass
    def test_roks_grad(self):
        pass

if __name__ == "__main__":
    unittest.main()
