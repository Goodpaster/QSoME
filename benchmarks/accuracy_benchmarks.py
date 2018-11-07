# Uses a test set of simple molecules to benchmark the accuracy results.
# By Daniel Graham

#Run like a unittests, but should not be run not on a node.

sys_1_react_fn = 'sys_1_react.inp'
system_1_react = '''
subsystem
C    0.1540 0.1385 0.2119
H    0.4642 0.2021 1.2505
H    1.0184 0.2323 -0.4367
H   -0.5732 0.9107 -0.0197
Cl  -0.6267 -1.4696 -0.0698
O    1.7911 2.5322 0.4194
O    2.8188 1.6681 -0.2141
H    3.3407 2.3516 -0.6428
charge -1
end

subsystem
O    1.9616 1.5620 2.7386
H    2.8538 1.2136 2.6868
H    1.8680 2.0002 1.7938
end

embed
 env_method pbe
 huzinaga
 cycles 250
end

basis 6-311g
active_method ccsd
grid 3
'''

sys_1_trans_fn = 'sys_1_trans.inp'
system_1_trans = '''
subsystem
C    0.0878 0.3745 -0.0150 
H   -0.7803 0.9753 0.1966 
H    1.0159 0.5957 0.4801 
H    0.1006 -0.2548 -0.8857
Cl  -0.4662 -1.2008 1.3248
O    0.5636 1.8657 -1.2969
O    1.8225 1.4130 -1.8994
H    1.5547 1.3877 -2.8233
charge -1
end

subsystem
O    1.5260 4.2554 -0.7329
H    2.4429 3.9976 -0.8456
H    1.0718 3.3814 -0.9107
end

embed
 env_method pbe
 huzinaga
 cycles 250
end

basis 6-311g
active_method ccsd
grid 3
'''

sys_2_react_fn = 'sys_2_react.inp'
system_2_react = '''
subsystem
F 2.4413 -0.3628 -0.0000
C 1.3216 0.4764 0.0000 
H 1.3921 1.1084 -0.8894 
H 1.3921 1.1084 0.8895 
C 0.0507 -0.3456 0.0000 
H 0.0535 -0.9973 0.8788
H 0.0536 -0.9973 -0.8788
charge -1
end

subsystem
C -1.2068 0.5264 -0.0000
H -1.1942 1.1840 -0.8754
H -1.1942 1.1841 0.8754
C -2.4966 -0.2923 0.0000
H -2.5557 -0.9346 0.8817
H -2.5556 -0.9347 -0.8817
H -3.3759 0.3545 -0.0000
charge +1
end

embed
 env_method m06
 huzinaga
 cycles 50
end

basis aug-cc-pVTZ
active_method ccsd
'''

sys_2_trans_fn = 'sys_2_trans.inp'
system_2_trans = '''
subsystem
Cl -2.1626 -0.8661 -0.0541 
F 0.3924 2.5800 -0.1257 
C -0.7289 0.7346 0.0630 
H -0.7810 0.9905 -0.9757 
H -1.3066 1.3495 0.7252 
C 0.5030 0.0655 0.6021 
H 0.2231 -0.6924 1.3400 
H 1.0602 0.8610 1.1005
charge -2
end

subsystem
C 1.3742 -0.5530 -0.4864
H 0.7866 -1.2885 -1.0453
H 1.6460 0.2476 -1.1792
C 2.6367 -1.2149 0.0638
H 3.2496 -0.4933 0.6121
H 2.3896 -2.0279 0.7538
H 3.2546 -1.6359 -0.7352
charge +1
end

embed
 env_method m06
 huzinaga
 cycles 50
end

basis aug-cc-pVTZ
active_method ccsd
'''

sys_3_react_fn = 'sys_3_react.inp'
system_3_react = '''
subsystem
C -0.2034 3.7809 -0.2543
O -1.1936 2.8460 -0.1702
H -2.0334 3.3226 -0.2712 
O -0.4414 4.9614 -0.4250
charge -1
end

subsystem
C 1.1296 3.1655 -0.1127
H 1.1664 2.0855 0.0352
C 2.2512 3.9148 -0.1647
C 3.5969 3.3849 -0.0335
C 4.7008 4.1530 -0.0881
H 2.1426 4.9929 -0.3156
H 5.6985 3.7260 0.0133
H 4.6346 5.2325 -0.2362
H 3.6963 2.3059 0.1165
charge +1
end

embed
 env_method m06
 huzinaga
 cycles 50
end

basis aug-cc-pVTZ
active_method ccsd
'''

sys_3_trans_fn = 'sys_3_trans.inp'
system_3_trans = '''
subsystem
C -0.2900 3.7344 -0.2511
O -1.1875 2.8480 -0.1707
O -0.4118 4.9767 -0.4238
charge -2
end

subsystem
C 1.1369 3.1937 -0.1160
H 1.2186 2.1104 0.0342
C 2.2543 3.9487 -0.1694
C 3.6014 3.4182 -0.0379
C 4.7421 4.1398 -0.0847
H 2.1281 5.0257 -0.3212
H 5.7212 3.6699 0.0225
H 4.7226 5.2222 -0.2320
H 3.6774 2.3344 0.1123
charge +1
end

embed
 env_method m06
 huzinaga
 cycles 50
end

basis aug-cc-pVTZ
active_method ccsd
'''

sys_4_react_fn = 'sys_4_react.inp'
system_4_react = '''
subsystem
F -6.3177 -0.0740 0.0000 
C -5.1535 0.6235 0.0000 
H -5.3201 1.7006 0.0000
end

subsystem
C -3.9681 -0.0010 0.0000
C -2.7049 0.7091 0.0000
H -3.9578 -1.0938 0.0000
C -1.4922 0.1041 0.0000
H -2.7469 1.8031 0.0000
C -0.2273 0.8164 0.0000
H -1.4453 -0.9899 0.0000
C 0.9820 0.2229 0.0000
H 1.9024 0.8062 0.0000
H 1.0832 -0.8643 0.0000
H -0.2818 1.9099 0.0000
end

embed
 env_method m06
 huzinaga
 cycles 50
end

basis aug-cc-pVTZ
active_method ccsd
'''

sys_4_trans_fn = 'sys_4_trans.inp'
system_4_trans = '''
subsystem
C -5.1535 0.6235 0.0000
H -5.3201 1.7006 0.0000
charge +1
end

subsystem
C -3.9681 -0.0010 0.0000
C -2.7049 0.7091 0.0000
H -3.9578 -1.0938 0.0000
C -1.4922 0.1041 0.0000
H -2.7469 1.8031 0.0000
C -0.2273 0.8164 0.0000
H -1.4453 -0.9899 0.0000
C 0.9820 0.2229 0.0000
H 1.9024 0.8062 0.0000
H 1.0832 -0.8643 0.0000
H -0.2818 1.9099 0.0000
end

embed
 env_method m06
 huzinaga
 cycles 50
end

basis aug-cc-pVTZ
active_method ccsd
'''

from qsome import inp_reader, cluster_supersystem, cluster_subsystem
from pyscf import gto, scf, dft, cc

import os
import sys
import shutil


base_path = os.path.dirname(os.path.realpath(__file__))
canonical_dir = "/canonical_energies/"
temp_dir = "/temp_inp/"
inp_path = base_path + temp_dir
canon_path = base_path + canonical_dir

if os.path.isdir(inp_path):
    shutil.rmtree(inp_path)    

os.mkdir(inp_path)

with open(inp_path+sys_1_react_fn, 'w') as f:
    f.write(system_1_react)

with open(inp_path+sys_1_trans_fn, 'w') as f:
    f.write(system_1_trans)

with open(inp_path+sys_2_react_fn, 'w') as f:
    f.write(system_2_react)

with open(inp_path+sys_2_trans_fn, 'w') as f:
    f.write(system_2_trans)

with open(inp_path+sys_3_react_fn, 'w') as f:
    f.write(system_3_react)

with open(inp_path+sys_3_trans_fn, 'w') as f:
    f.write(system_3_trans)

with open(inp_path+sys_4_react_fn, 'w') as f:
    f.write(system_4_react)

with open(inp_path+sys_4_trans_fn, 'w') as f:
    f.write(system_4_trans)


if not os.path.isdir(canon_path):
    os.mkdir(canon_path)

#Tests system 1
subsystems = []
in_obj = inp_reader.InpReader(inp_path + sys_1_react_fn)
for i in range(len(in_obj.subsys_mols)):
    if i == 0:
        mol = in_obj.subsys_mols[i]
        env_method = in_obj.env_subsystem_kwargs[i].pop('env_method')
        env_kwargs = in_obj.env_subsystem_kwargs[i]
        active_method = in_obj.active_subsystem_kwargs.pop('active_method')
        active_kwargs = in_obj.active_subsystem_kwargs
        active_kwargs.update(env_kwargs)
        subsys = cluster_subsystem.ClusterActiveSubSystem(mol, env_method, active_method,  **active_kwargs)
        subsystems.append(subsys)
    else:
        mol = in_obj.subsys_mols[i]
        env_method = in_obj.env_subsystem_kwargs[i].pop('env_method')
        env_kwargs = in_obj.env_subsystem_kwargs[i]
        subsys = cluster_subsystem.ClusterEnvSubSystem(mol, env_method, **env_kwargs)
        subsystems.append(subsys)
ct_method = in_obj.supersystem_kwargs.pop('ct_method')
supersystem_kwargs = in_obj.supersystem_kwargs
supersystem = cluster_supersystem.ClusterSuperSystem(subsystems, 
    ct_method, **supersystem_kwargs)
supersystem.freeze_and_thaw()
supersystem.env_in_env_energy()
supersystem.get_active_energy()

sys_1_react_embed = supersystem.get_supersystem_energy() - supersystem.subsystems[0].env_energy + supersystem.subsystems[0].active_energy

try:
    with open(canon_path + os.path.splitext(sys_1_react_fn)[0] + '.out', 'r') as f:
        sys_1_react_canon = float(f.read())
except:
    mf = scf.RHF(supersystem.mol)
    hf_e = mf.kernel()
    ccsd_scf = cc.CCSD(mf)
    sys_1_react_canon = hf_e + ccsd_scf.kernel()[0]
    with open(canon_path + os.path.splitext(sys_1_react_fn)[0] + '.out', 'w') as f:
        f.write(str(sys_1_react_canon))
        
subsystems = []
in_obj = inp_reader.InpReader(inp_path + sys_1_trans_fn)
for i in range(len(in_obj.subsys_mols)):
    if i == 0:
        mol = in_obj.subsys_mols[i]
        env_method = in_obj.env_subsystem_kwargs[i].pop('env_method')
        env_kwargs = in_obj.env_subsystem_kwargs[i]
        active_method = in_obj.active_subsystem_kwargs.pop('active_method')
        active_kwargs = in_obj.active_subsystem_kwargs
        active_kwargs.update(env_kwargs)
        subsys = cluster_subsystem.ClusterActiveSubSystem(mol, env_method, active_method,  **active_kwargs)
        subsystems.append(subsys)
    else:
        mol = in_obj.subsys_mols[i]
        env_method = in_obj.env_subsystem_kwargs[i].pop('env_method')
        env_kwargs = in_obj.env_subsystem_kwargs[i]
        subsys = cluster_subsystem.ClusterEnvSubSystem(mol, env_method, **env_kwargs)
        subsystems.append(subsys)
ct_method = in_obj.supersystem_kwargs.pop('ct_method')
supersystem_kwargs = in_obj.supersystem_kwargs
supersystem = cluster_supersystem.ClusterSuperSystem(subsystems, 
    ct_method, **supersystem_kwargs)
supersystem.freeze_and_thaw()
supersystem.env_in_env_energy()
supersystem.get_active_energy()

sys_1_trans_embed = supersystem.get_supersystem_energy() - supersystem.subsystems[0].env_energy + supersystem.subsystems[0].active_energy

try:
    with open(canon_path + os.path.splitext(sys_1_trans_fn)[0] + '.out', 'r') as f:
        sys_1_trans_canon = float(f.read())
except:
    mf = scf.RHF(supersystem.mol)
    hf_e = mf.kernel()
    ccsd_scf = cc.CCSD(mf)
    sys_1_trans_canon = hf_e + ccsd_scf.kernel()[0]
    with open(canon_path + os.path.splitext(sys_1_trans_fn)[0] + '.out', 'w') as f:
        f.write(str(sys_1_trans_canon))

emb_diff = (sys_1_trans_embed - sys_1_react_embed) * 627.5
canon_diff = (sys_1_trans_canon - sys_1_react_canon) * 627.5
err = emb_diff - canon_diff
print (f"Embedding Test 1 Difference: {emb_diff} kcal/mol")
print (f"Canonical Test 1 Difference: {canon_diff} kcal/mol")
print (f"Test 1 Error: {err} kcal/mol")


