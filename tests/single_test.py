
# Uses a test set of simple molecules to benchmark the accuracy results.
# By Daniel Graham

#Run like a unittests, but should not be run not on a node.

he_fn = 'he.inp'
system_1_react = '''
subsystem
He          0.00000       0.00000       0.00000
gh.He          1.5000       0.00000       0.00000
end

subsystem
He          1.5000       0.00000       0.00000
gh.He          0.00000       0.00000       0.00000
end

embed
 env_method pbe
 huzinaga
 cycles 20
end

basis 6-31g
active_method hf
'''

from qsome import inp_reader, cluster_supersystem, cluster_subsystem
from pyscf import gto, scf, dft, cc

import os
import sys
import shutil


base_path = os.path.dirname(os.path.realpath(__file__))
temp_dir = "/he_tmep/"
inp_path = base_path + temp_dir

if os.path.isdir(inp_path):
    shutil.rmtree(inp_path)    

os.mkdir(inp_path)

with open(inp_path+he_fn, 'w') as f:
    f.write(system_1_react)

#Tests system 1
subsystems = []
in_obj = inp_reader.InpReader(inp_path + he_fn)
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

print (f"Embedding He: {sys_1_react_embed} kcal/mol")
