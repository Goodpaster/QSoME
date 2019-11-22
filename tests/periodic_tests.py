# Tests for the input reader object
# Dhabih V. Chulhai


import unittest
import os
import shutil
import re
import numpy as np

from qsome import inp_reader, periodic_subsystem, periodic_supersystem
from pyscf import gto 

minimal_input = """
subsystem
He     15.000       2.000       2.000
end

subsystem
He      5.000       2.000       2.000
He     25.000       2.000       2.000
end

active_method LDA,VWN

embed
 env_method LDA,VWN
end

basis
 default 631g
end

periodic_settings
 lattice_vectors
  30.000 0.000 0.000
   0.000 4.000 0.000
   0.000 0.000 4.000
 end 

 dimensions 1
 kpoints 1 1 1 
end
"""

supersystem_basis_input = """
subsystem
Gh.He  0.5000       2.000       2.000
He     1.5000       2.000       2.000
Gh.He  2.5000       2.000       2.000
end

subsystem
He     0.5000       2.000       2.000
Gh.He  1.5000       2.000       2.000
He     2.5000       2.000       2.000
end

active_method LDA,VWN

embed
 env_method LDA,VWN
end

basis
 default 631g
end

periodic_settings
 lattice_vectors
   3.000 0.000 0.000
   0.000 4.000 0.000
   0.000 0.000 4.000
 end 

 dimensions 1
 kpoints 2 1 1 
end
"""


he_chain_input = """
#
subsystem
He     2.5000       2.000       2.000
end

subsystem
He     0.5000       2.000       2.000
He     1.5000       2.000       2.000
He     3.5000       2.000       2.000
He     4.5000       2.000       2.000
end

active_method ccsd

embed
 huzinaga
 env_method LDA,VWN
 cycles 30
end

basis
 default 631g
end

periodic_settings
 lattice_vectors
   5.000 0.000 0.000
   0.000 4.000 0.000
   0.000 0.000 4.000
 end 

 dimensions 1
 kpoints 3 1 1 
 precision 1e-6
 exp_to_discard 0.1
 density_fit mdf
 low_dim_ft_type inf_vacuum
end
"""


class TestBaseClass():

    def prepare_input(self, input_str):
        self.path = os.getcwd() + "/temp_input/"
        self.filename = filename = "default.inp"
        if os.path.isdir(self.path):
            shutil.rmtree(self.path)
        os.mkdir(self.path)

        with open(self.path + self.filename, "w") as f:
            f.write(input_str)


    def run_embedding(self, active=True):
        if not hasattr(self, 'kpts'):
            self.kpts, self.nkpts = periodic_subsystem.InitKpoints(self.in_obj.subsys_mols[0],
                                                     **self.in_obj.kpoints_kwargs) 

        env_method = self.in_obj.env_subsystem_kwargs[0].pop('env_method')
        subsysA = periodic_subsystem.PeriodicEnvSubSystem(self.in_obj.subsys_mols[0],
                    env_method, self.kpts, **self.in_obj.periodic_kwargs)

        env_method = self.in_obj.env_subsystem_kwargs[1].pop('env_method')
        subsysB = periodic_subsystem.PeriodicEnvSubSystem(self.in_obj.subsys_mols[1],
                    env_method, self.kpts, **self.in_obj.periodic_kwargs)

        subsystems = [subsysA, subsysB]

        fs_method = self.in_obj.supersystem_kwargs.pop('fs_method')

        supersystem = periodic_supersystem.PeriodicSuperSystem(subsystems, env_method,
            kpoints=self.kpts, **self.in_obj.cell_kwargs, **self.in_obj.periodic_kwargs,
            **self.in_obj.supersystem_kwargs)

        super_energy = supersystem.get_supersystem_energy()

        supersystem.freeze_and_thaw()
        supersystem.env_in_env_energy()

        if active:
            active_method = self.in_obj.active_subsystem_kwargs.pop('active_method')
            active_kwargs = self.in_obj.active_subsystem_kwargs
            supersystem.periodic_to_cluster(active_method, **active_kwargs)

            supersystem.get_active_energy()

        return supersystem


    def tearDown(self):
        if os.path.isdir(self.path):
            shutil.rmtree(self.path)


class TestPeriodicInput(unittest.TestCase, TestBaseClass):

    def setUp(self):
        self.prepare_input(minimal_input)
        self.in_obj = inp_reader.InpReader(self.path + self.filename)


    def test_dimension(self):
        self.assertEqual(self.in_obj.cell_kwargs['dimension'], 1)


    def test_periodic_input(self):
        self.assertTrue(self.in_obj.periodic)


    def test_lattice_parameters(self):
        lattice_A = np.array([[30., 0, 0], [0, 4, 0], [0, 0, 4]])
        lattice_B = np.array([[56.69178374,  0.        ,  0.        ],
                            [ 0.        ,  7.5589045 ,  0.        ],  
                            [ 0.        ,  0.        ,  7.5589045 ]])

        for icell in range(2):
            cell = self.in_obj.subsys_mols[icell]
            a = cell.lattice_vectors()

            for i in range(3):
                for j in range(3):
                    a_input = self.in_obj.cell_kwargs['a'][i,j]
                    b_input = a[i,j]

                    self.assertAlmostEqual(a_input, lattice_A[i,j])
                    self.assertAlmostEqual(b_input, lattice_B[i,j])


    def test_kpoints(self):
        for i in range(3):
            self.assertEqual(self.in_obj.kpoints_kwargs['kpoints'][i], 1)

        if not hasattr(self, 'kpts'):
            self.kpts, self.nkpts = periodic_subsystem.InitKpoints(self.in_obj.subsys_mols[0],
                                        **self.in_obj.kpoints_kwargs)
        self.assertEqual(self.nkpts, 1)
        for i in range(3):
            self.assertEqual(self.kpts[0,i], 0.0)


    def test_isolated_energies(self):
        supersystem = self.run_embedding()
        self.assertAlmostEqual(supersystem.subsystems[0].periodic_in_periodic_env_energy,
            supersystem.subsystems[0].cluster_in_periodic_env_energy, 5)
        self.assertAlmostEqual(supersystem.subsystems[0].cluster_in_periodic_env_energy,
            supersystem.subsystems[0].active_energy)


class TestSuperSystemBasis(unittest.TestCase, TestBaseClass):

    def setUp(self):
        self.prepare_input(supersystem_basis_input)
        self.in_obj = inp_reader.InpReader(self.path + self.filename)

    def test_freeze_and_thaw_energy(self):
        supersystem = self.run_embedding(active=False)
        self.assertAlmostEqual(supersystem.fs_energy, supersystem.ft_energy)


class TestHeChainEnergies(unittest.TestCase, TestBaseClass):

    def setUp(self):
        self.prepare_input(he_chain_input)
        self.in_obj = inp_reader.InpReader(self.path + self.filename)    

    def test_cluster_approximation(self):
        supersystem = self.run_embedding()
        self.assertAlmostEqual(supersystem.subsystems[0].periodic_in_periodic_env_energy,
            supersystem.subsystems[0].cluster_in_periodic_env_energy, 3)


if __name__ == "__main__":
    unittest.main()
