import unittest
import os
import shutil
import re

from qsome import inp_reader, cluster_subsystem, cluster_supersystem
from pyscf import gto

import numpy as np

#TODO This could use more testing for open shell systems and the like.

def_filename = "default.inp"
default_str = """
subsystem
He    0.0000    0.0000    0.0000
end

subsystem
He    2.0000    0.0000    0.0000
end

embed
 env_method pbe
 huzinaga
end

basis 3-21g
active_method hf
"""

partial_ghost_filename = 'partial_ghost.inp'
partial_ghost_str = default_str.replace("""
He    0.0000    0.0000    0.0000
""","""
He    0.0000    0.0000    0.0000
gh.He    2.0000    0.0000    0.0000
""", 1)

ghost_filename = 'ghost.inp'
ghost_str = partial_ghost_str.replace("""
subsystem
He    2.0000    0.0000    0.0000
""","""
subsystem
He    2.0000    0.0000    0.0000
gh.He    0.0000    0.0000    0.0000
""", 1)

widesep_filename = 'wide_sep.inp'
widesep_str = default_str.replace("""
He    2.0000    0.0000    0.0000
""", """
He    200.0000    0.0000    0.0000
""", 1)

temp_inp_dir = "/temp_input/"

class TestSuperSystemDensity(unittest.TestCase):

    def setUp(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way

        if os.path.isdir(path):
            shutil.rmtree(path)    
        os.mkdir(path)

        with open(path+def_filename, 'w') as f:
            f.write(default_str)

        with open(path+ghost_filename, 'w') as f:
            f.write(ghost_str)

        with open(path+widesep_filename, 'w') as f:
            f.write(widesep_str)

    def test_pyscf_init(self):
        subsystems = []
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        in_obj = inp_reader.InpReader(path + widesep_filename)
        for i in range(len(in_obj.subsys_mols)):
            mol = in_obj.subsys_mols[i]
            env_method = in_obj.env_subsystem_kwargs[i].pop('env_method')
            env_kwargs = in_obj.env_subsystem_kwargs[i]
            subsys = cluster_subsystem.ClusterEnvSubSystem(mol, env_method, **env_kwargs)
            subsystems.append(subsys)
        ct_method = in_obj.supersystem_kwargs.pop('ct_method')
        supersystem_kwargs = in_obj.supersystem_kwargs
        supersystem = cluster_supersystem.ClusterSuperSystem(subsystems, 
            ct_method, **supersystem_kwargs)

        sup_scf = supersystem.ct_scf
        if supersystem.initguess != None:
            pyscf_dmat = sup_scf.get_init_guess(key=supersystem.initguess)
        else:
            pyscf_dmat = sup_scf.get_init_guess()
        self.assertTrue(np.array_equal(supersystem.dmat[0] + supersystem.dmat[1], pyscf_dmat))

        if subsystems[0].initguess != None:
            pyscf_dmat = subsystem[0].env_scf.get_init_guess(key=subsystems[0].initguess)
        else:
            pyscf_dmat = subsystems[0].env_scf.get_init_guess()
        self.assertTrue(np.array_equal(subsystems[0].dmat[0] + subsystems[0].dmat[1], pyscf_dmat))

        if subsystems[1].initguess != None:
            pyscf_dmat = subsystems[1].env_scf.get_init_guess(key=subsystems[1].initguess)
        else:
            pyscf_dmat = subsystems[1].env_scf.get_init_guess()

        self.assertTrue(np.array_equal(subsystems[1].dmat[0] + subsystems[1].dmat[1], pyscf_dmat))
         
         
    def test_supmol_init(self):
        subsystems = []
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        in_obj = inp_reader.InpReader(path + def_filename)
        for i in range(len(in_obj.subsys_mols)):
            mol = in_obj.subsys_mols[i]
            env_method = in_obj.env_subsystem_kwargs[i].pop('env_method')
            in_obj.env_subsystem_kwargs[i]['initguess'] = 'supmol'
            env_kwargs = in_obj.env_subsystem_kwargs[i]
            subsys = cluster_subsystem.ClusterEnvSubSystem(mol, env_method, **env_kwargs)
            subsystems.append(subsys)
        ct_method = in_obj.supersystem_kwargs.pop('ct_method')
        in_obj.supersystem_kwargs['initguess'] = 'supmol'
        supersystem_kwargs = in_obj.supersystem_kwargs
        supersystem = cluster_supersystem.ClusterSuperSystem(subsystems, 
            ct_method, **supersystem_kwargs)

        sup_scf = supersystem.ct_scf
        sup_scf.kernel()
        pyscf_dmat = sup_scf.make_rdm1()
        self.assertTrue(np.array_equal(supersystem.dmat[0] + supersystem.dmat[1], pyscf_dmat))

    def test_readchk_init(self):
        subsystems = []
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        in_obj = inp_reader.InpReader(path + def_filename)
        for i in range(len(in_obj.subsys_mols)):
            mol = in_obj.subsys_mols[i]
            env_method = in_obj.env_subsystem_kwargs[i].pop('env_method')
            in_obj.env_subsystem_kwargs[i]['initguess'] = 'supmol'
            env_kwargs = in_obj.env_subsystem_kwargs[i]
            subsys = cluster_subsystem.ClusterEnvSubSystem(mol, env_method, **env_kwargs)
            subsystems.append(subsys)
        ct_method = in_obj.supersystem_kwargs.pop('ct_method')
        in_obj.supersystem_kwargs['initguess'] = 'supmol'
        supersystem_kwargs = in_obj.supersystem_kwargs
        supersystem = cluster_supersystem.ClusterSuperSystem(subsystems, 
            ct_method, **supersystem_kwargs)

        old_dmat = np.copy(supersystem.dmat)
        old_dmat_1 = np.copy(subsystems[0].dmat)
        old_dmat_2 = np.copy(subsystems[1].dmat)

        subsystems = []
        for i in range(len(in_obj.subsys_mols)):
            mol = in_obj.subsys_mols[i]
            in_obj.env_subsystem_kwargs[i]['initguess'] = 'readchk'
            env_kwargs = in_obj.env_subsystem_kwargs[i]
            subsys = cluster_subsystem.ClusterEnvSubSystem(mol, env_method, **env_kwargs)
            subsystems.append(subsys)

        in_obj.supersystem_kwargs['initguess'] = 'readchk'
        in_obj.supersystem_kwargs['ft_initguess'] = 'readchk'
        supersystem_kwargs = in_obj.supersystem_kwargs
        supersystem = cluster_supersystem.ClusterSuperSystem(subsystems, 
            ct_method, **supersystem_kwargs)

        self.assertTrue(np.array_equal(supersystem.dmat, old_dmat))
        self.assertTrue(np.array_equal(supersystem.subsystems[0].dmat, old_dmat_1))
        self.assertTrue(np.array_equal(supersystem.subsystems[1].dmat, old_dmat_2))


    def tearDown(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way.
        if os.path.isdir(path):
            shutil.rmtree(path)    

if __name__ == "__main__":
    unittest.main()
