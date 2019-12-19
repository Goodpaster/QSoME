from pyscf import gto
import time
from qsome import interaction_mediator, cluster_subsystem, cluster_supersystem

from copy import deepcopy as copy

#All mol objects which will be used for testing.

sn2_mol1 = gto.M()
sn2_mol1.atom = '''
Cl         3.51109       -0.30375        0.00020
C          1.97915        0.64190        0.00018
H          2.00892        1.27702       -0.88401
H          2.00850        1.27637        0.88486
'''

sn2_mol2 = gto.M()
sn2_mol2.atom = '''
C          0.77452       -0.25849       -0.00044
H          0.80952       -0.91242       -0.87692
H          0.80921       -0.91317        0.87550
C         -0.51363        0.54349       -0.00030
H         -0.53565        1.20393        0.87514
H         -0.53562        1.20434       -0.87545
C         -1.75043       -0.33137       -0.00050
H         -1.72894       -0.99275       -0.87536
H         -1.72864       -0.99348        0.87379
C         -3.04233        0.46077        0.00008
H         -3.05786        1.12092        0.87408
H         -3.05853        1.12117       -0.87372
C         -4.26820       -0.42758        0.00047
H         -4.28002       -1.07524        0.88008
H         -5.19301        0.14965        0.00102
H         -4.28083       -1.07487       -0.87940
'''

sn2_mol3 = gto.M()
sn2_mol3.atom = '''
Cl         3.51109       -0.30375        0.00020
C          1.97915        0.64190        0.00018
H          2.00892        1.27702       -0.88401
H          2.00850        1.27637        0.88486
C          0.77452       -0.25849       -0.00044
H          0.80952       -0.91242       -0.87692
H          0.80921       -0.91317        0.87550
C         -0.51363        0.54349       -0.00030
H         -0.53565        1.20393        0.87514
H         -0.53562        1.20434       -0.87545
'''

sn2_mol4 = gto.M()
sn2_mol4.atom = '''
C         -1.75043       -0.33137       -0.00050
H         -1.72894       -0.99275       -0.87536
H         -1.72864       -0.99348        0.87379
C         -3.04233        0.46077        0.00008
H         -3.05786        1.12092        0.87408
H         -3.05853        1.12117       -0.87372
C         -4.26820       -0.42758        0.00047
H         -4.28002       -1.07524        0.88008
H         -5.19301        0.14965        0.00102
H         -4.28083       -1.07487       -0.87940
'''

bond_cleave1 = gto.M()
bond_cleave1.atom = '''
C          0.84878        1.53039        0.00000
H          1.50962        1.58000        0.87454
H          1.50962        1.58000       -0.87454
C         -0.84878       -1.53039        0.00000
H         -1.50962       -1.58000       -0.87454
H         -1.50962       -1.58000        0.87454
'''

bond_cleave2 = gto.M()
bond_cleave2.atom = '''
C         -0.46985        7.76776        0.00000
H          0.06673        8.71696        0.00000
H         -1.11758        7.75217        0.87964
H         -1.11758        7.75217       -0.87964
C          0.46985        6.58062        0.00000
H          1.12915        6.62528        0.87377
H          1.12915        6.62528       -0.87377
C         -0.26441        5.25506        0.00000
H         -0.92565        5.20629        0.87450
H         -0.92565        5.20629       -0.87450
C          0.65932        4.05507        0.00000
H          1.32017        4.10471        0.87457
H          1.32017        4.10471       -0.87457
C         -0.07512        2.73049        0.00000
H         -0.73597        2.68085       -0.87453
H         -0.73597        2.68085        0.87453
C          0.07512       -2.73049        0.00000
H          0.73597       -2.68085        0.87453
H          0.73597       -2.68085       -0.87453
C         -0.65932       -4.05507        0.00000
H         -1.32017       -4.10471       -0.87457
H         -1.32017       -4.10471        0.87457
C          0.26441       -5.25506        0.00000
H          0.92565       -5.20629        0.87450
H          0.92565       -5.20629       -0.87450
C         -0.46985       -6.58062        0.00000
H         -1.12915       -6.62528       -0.87377
H         -1.12915       -6.62528        0.87377
C          0.46985       -7.76776        0.00000
H          1.11758       -7.75217        0.87964
H         -0.06673       -8.71696        0.00000
H          1.11758       -7.75217       -0.87964
'''

bond_cleave3 = gto.M()
bond_cleave3.atom = '''
C          0.65932        4.05507        0.00000
H          1.32017        4.10471        0.87457
H          1.32017        4.10471       -0.87457
C         -0.07512        2.73049        0.00000
H         -0.73597        2.68085       -0.87453
H         -0.73597        2.68085        0.87453
C          0.84878        1.53039        0.00000
H          1.50962        1.58000        0.87454
H          1.50962        1.58000       -0.87454
C         -0.84878       -1.53039        0.00000
H         -1.50962       -1.58000       -0.87454
H         -1.50962       -1.58000        0.87454
C          0.07512       -2.73049        0.00000
H          0.73597       -2.68085        0.87453
H          0.73597       -2.68085       -0.87453
C         -0.65932       -4.05507        0.00000
H         -1.32017       -4.10471       -0.87457
H         -1.32017       -4.10471        0.87457
'''

bond_cleave4 = gto.M()
bond_cleave4.atom = '''
C         -0.46985        7.76776        0.00000
H          0.06673        8.71696        0.00000
H         -1.11758        7.75217        0.87964
H         -1.11758        7.75217       -0.87964
C          0.46985        6.58062        0.00000
H          1.12915        6.62528        0.87377
H          1.12915        6.62528       -0.87377
C         -0.26441        5.25506        0.00000
H         -0.92565        5.20629        0.87450
H         -0.92565        5.20629       -0.87450
C          0.26441       -5.25506        0.00000
H          0.92565       -5.20629        0.87450
H          0.92565       -5.20629       -0.87450
C         -0.46985       -6.58062        0.00000
H         -1.12915       -6.62528       -0.87377
H         -1.12915       -6.62528        0.87377
C          0.46985       -7.76776        0.00000
H          1.11758       -7.75217        0.87964
H         -0.06673       -8.71696        0.00000
H          1.11758       -7.75217       -0.87964
'''

thiol1 = gto.M()
thiol1.atom = '''
C         -1.39013        0.60272        0.00001
H         -1.36123        1.27811       -0.87485
H         -1.36123        1.27809        0.87489
S         -3.02131       -0.17073       -0.00001
'''

thiol2 = gto.M()
thiol2.atom = '''
C          3.63982        0.17268       -0.00001
H          4.49175       -0.52133        0.00002
H          3.73370        0.81761        0.88742
H          3.73370        0.81754       -0.88749
C          2.31212       -0.56204        0.00001
H          2.24759       -1.22456        0.88028
H          2.24758       -1.22460       -0.88022
C          1.11974        0.37903       -0.00001
H          1.17755        1.04420        0.88131
H          1.17756        1.04418       -0.88134
C         -0.21248       -0.35144       -0.00000
H         -0.28014       -1.01161        0.88083
H         -0.28015       -1.01160       -0.88084
'''

thiol3 = gto.M()
thiol3.atom = '''
C         -1.39013        0.60272        0.00001
H         -1.36123        1.27811       -0.87485
H         -1.36123        1.27809        0.87489
S         -3.02131       -0.17073       -0.00001
C          1.11974        0.37903       -0.00001
H          1.17755        1.04420        0.88131
H          1.17756        1.04418       -0.88134
C         -0.21248       -0.35144       -0.00000
H         -0.28014       -1.01161        0.88083
H         -0.28015       -1.01160       -0.88084
'''

thiol4 = gto.M()
thiol4.atom = '''
C          3.63982        0.17268       -0.00001
H          4.49175       -0.52133        0.00002
H          3.73370        0.81761        0.88742
H          3.73370        0.81754       -0.88749
C          2.31212       -0.56204        0.00001
H          2.24759       -1.22456        0.88028
H          2.24758       -1.22460       -0.88022
'''

mof1 = gto.M()
mof1.atom = '''
Fe         0.00000       -0.00000        0.05961
'''

mof2 = gto.M()
mof2.atom = '''
O          1.38352        1.42088       -0.29729
O          1.38352       -1.42088       -0.29729
O         -1.38352       -1.42088       -0.29729
C          1.21957       -2.62333       -0.64080
C         -0.00000       -3.27203       -0.81174
C         -1.21957       -2.62333       -0.64080
O         -1.38352        1.42088       -0.29729
C         -1.21957        2.62333       -0.64080
H         -2.13506       -3.20415       -0.82549
H          2.13506       -3.20415       -0.82549
C          1.21957        2.62333       -0.64080
C          0.00000        3.27203       -0.81174
H          2.13506        3.20415       -0.82549
H         -2.13506        3.20415       -0.82549
C          0.00000       -0.00000        2.11025
O          0.00000       -0.00000        3.24357
H         -0.00000        4.30911       -1.11232
H         -0.00000       -4.30911       -1.11232
'''

mof3 = gto.M()
mof3.atom = '''
Fe         0.00000       -0.00000        0.05961
O          1.38352        1.42088       -0.29729
O          1.38352       -1.42088       -0.29729
O         -1.38352       -1.42088       -0.29729
O         -1.38352        1.42088       -0.29729
'''

mof4 = gto.M()
mof4.atom = '''
C          1.21957       -2.62333       -0.64080
C         -0.00000       -3.27203       -0.81174
C         -1.21957       -2.62333       -0.64080
C         -1.21957        2.62333       -0.64080
H         -2.13506       -3.20415       -0.82549
H          2.13506       -3.20415       -0.82549
C          1.21957        2.62333       -0.64080
C          0.00000        3.27203       -0.81174
H          2.13506        3.20415       -0.82549
H         -2.13506        3.20415       -0.82549
C          0.00000       -0.00000        2.11025
O          0.00000       -0.00000        3.24357
H         -0.00000        4.30911       -1.11232
H         -0.00000       -4.30911       -1.11232
'''

mol_list = [sn2_mol1, sn2_mol2, sn2_mol3, sn2_mol4, bond_cleave1, bond_cleave2, bond_cleave3, bond_cleave4, thiol1, thiol2, thiol3, thiol4, mof1, mof2, mof3, mof4]
basis_set_list = ['3-21g', '6-31g', 'cc-pVDZ', 'aug-cc-pVDZ', 'cc-pVTZ']
charges_cs = [-1, 1, -1, 1, -2, 2, -2, 2, -2, 1, -2, 1, 2, -2, -6, 6]
charges_os = [0, 0, 0, 0, 0, 0, 0, 0, -1, 1, -1, 1, 2, -2, -6, 6]
spin_os = [1, -1, 1, -1, 0, 0, 0, 0, 1, 0, 1, 0, 4, 0, 4, 0]

grid_points = [2,3,4,5]
init_guess = ['atom', 'h1e']#, 'super', 'sub', 'localsuper']
xc_fun = ['lda', 'pbe', 'b3lyp', 'm06']

class MolObjects:
    def __iter__(self):
        self.mol_index = 0
        self.basis_index = 0
        self.charges_index = 0
        self.grid_index = 0
        mol1 = mol_list[0]
        mol1.basis = basis_set_list[0]
        mol1.charge = charges_cs[0]

        mol2 = mol_list[1]
        mol2.basis = basis_set_list[0]
        mol2.charge = charges_cs[1]
        mol1.build()
        mol2.build()
        self.subs = (mol1, mol2)
        return self

    def __next__(self):
        temp1 = gto.mole.copy(self.subs[0])
        temp2 = gto.mole.copy(self.subs[1])
        x = (temp1, temp2)
        self.mol_index += 2
        if self.mol_index >= len(mol_list):
            self.mol_index = 0
            self.basis_index += 1
            if self.basis_index >= len(basis_set_list):
                self.basis_index = 0
                self.charges_index += 1
                if self.charges_index >= 2:
                    return None
        mol1 = mol_list[self.mol_index]
        mol1.basis = basis_set_list[self.basis_index]
        if self.charges_index > 0:
            mol1.charge = charges_os[self.mol_index]
            mol1.spin = spin_os[self.mol_index]
        else:
            mol1.charge = charges_cs[self.mol_index]

        mol2 = mol_list[self.mol_index + 1]
        mol2.basis = basis_set_list[self.basis_index]
        if self.charges_index > 0:
            mol2.charge = charges_os[self.mol_index + 1]
            mol2.spin = spin_os[self.mol_index + 1]
        else:
            mol2.charge = charges_cs[self.mol_index + 1]

        mol1.build()
        mol2.build()
        self.subs = (mol1, mol2)
        return x

def density_damping():
    output_filename = 'density_damping_results.out'
    tempfile = "temp.out"
    molO = MolObjects()
    moliter = iter(molO)
    x = next(moliter)
    num = 0
    while x is not None:
        for gp in grid_points:
            for ig in init_guess:
                for xc in xc_fun:
                    for fud in range(2):
                        header_string = f"molnum: {num}\nbasis: {x[0].basis}\ncharge: {x[0].charge}\ngridsize: {gp}\ninitguess: {ig}\nxc_fun: {xc}\nfock_update: {fud}\n"
                        with open(output_filename, 'a') as fout:
                            fout.write(header_string)
                        sub1 = cluster_subsystem.ClusterEnvSubSystem(x[0], xc)
                        sub2 = cluster_subsystem.ClusterEnvSubSystem(x[1], xc)
                        sup = cluster_supersystem.ClusterSuperSystem([sub1, sub2], xc, fs_grid_level=gp, ft_cycles=1000, ft_initguess=ig, ft_updatefock=fud)
                        sup.init_density()
                        start_time = time.time()
                        sup.freeze_and_thaw()
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        cycles = sup.ft_iter
                        write_string = f"  FT cycles: {cycles}\n  Elapsed Time: {elapsed_time}\n  Average time per cycle: {elapsed_time/float(cycles)}\n  Sub1 E: {sub1.get_env_energy()}\n  Sub2 E: {sub2.get_env_energy()}\n\n"
                        with open(output_filename, 'a') as fout:
                            fout.write(write_string)
                        if x[0].spin != 0 or x[1].spin != 0:
                            sub1 = cluster_subsystem.ClusterEnvSubSystem(x[0], xc, unrestricted=True)
                            sub2 = cluster_subsystem.ClusterEnvSubSystem(x[1], xc, unrestricted=True)
                            sup = cluster_supersystem.ClusterSuperSystem([sub1, sub2], xc, fs_grid_level=gp, ft_cycles=1000, ft_initguess=ig, ft_updatefock=fud, fs_unrestricted=True, ft_unrestricted=True)
                            sup.init_density()
                            start_time = time.time()
                            sup.freeze_and_thaw()
                            end_time = time.time()
                            elapsed_time = end_time - start_time
                            cycles = sup.ft_iter
                            write_string = f"Unrestricted\n  FT cycles: {cycles}\n  Elapsed Time: {elapsed_time}\n  Average time per cycle: {elapsed_time/float(cycles)}\n  Sub1 E: {sub1.get_env_energy()}\n  Sub2 E: {sub2.get_env_energy()}\n\n"

        num += 2
        x = next(moliter) 
        print ("Progress")
        print (f"{num/2} Done Total")
    
density_damping()


