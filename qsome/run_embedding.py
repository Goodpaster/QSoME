# A module to run an input file and get final embedding results ouput to stdout
# Daniel Graham

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from textwrap import dedent
import sys
import os

import inp_reader, cluster_subsystem, cluster_supersystem

from cluster_supersystem import time_method

@time_method("Total Embedding Time")
def main(filename):
    in_obj = inp_reader.InpReader(filename)
    subsystems = []
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
    #supersystem.env_in_env_energy()
    supersystem.get_active_energy()

    total_energy = supersystem.get_supersystem_energy() - supersystem.subsystems[0].env_energy + supersystem.subsystems[0].active_energy

    print("".center(80, '*'))
    print(f"Total Embedding Energy:     {total_energy}")
    print("".center(80,'*'))

    return total_energy


if __name__=='__main__':

    fullpath = os.path.abspath(sys.argv[1])
    main(fullpath)

