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
diis 6
initguess submol
end

subsystem
O    1.9616 1.5620 2.7386
H    2.8538 1.2136 2.6868
H    1.8680 2.0002 1.7938
diis 6
initguess submol
end

embed
 env_method m06
 huzinaga
 diis 0
 updatefock 1
 cycles 250
end

basis 6-311g
active_method ccsd
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
diis 6
initguess submol
end

subsystem
O    1.5260 4.2554 -0.7329
H    2.4429 3.9976 -0.8456
H    1.0718 3.3814 -0.9107
diis 6
initguess submol
end

embed
 env_method m06
 huzinaga
 cycles 250
 diis 0
 updatefock 1
end

basis 6-311g
active_method ccsd
'''

import os
from qsome import inp_reader, cluster_subsystem, cluster_supersystem
import sys
import shutil


base_path = os.path.dirname(os.path.realpath(__file__))
canonical_dir = "/canonical_energies/"
temp_dir = "/temp_inp_12/"
inp_path = base_path + temp_dir
canon_path = base_path + canonical_dir

if os.path.isdir(inp_path):
    shutil.rmtree(inp_path)    

os.mkdir(inp_path)

with open(inp_path+sys_1_react_fn, 'w') as f:
    f.write(system_1_react)

with open(inp_path+sys_1_trans_fn, 'w') as f:
    f.write(system_1_trans)

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


