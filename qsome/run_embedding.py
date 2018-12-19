# A module to run an input file and get final embedding results ouput to stdout
# Daniel Graham

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from textwrap import dedent
import sys
import os
from os.path import expanduser, expandvars, abspath, splitext

import inp_reader, cluster_subsystem, cluster_supersystem


def main():

    run_args = arguments()
    file_path = abspath(expandvars(expanduser(run_args.inp_file)))
    nproc = run_args.ppn
    pmem = run_args.pmem
    scr_dir = run_args.scr_dir

    in_obj = inp_reader.InpReader(file_path)
    subsystems = []
    for i in range(len(in_obj.subsys_mols)):
        if i == 0:
            mol = in_obj.subsys_mols[i]
            env_method = in_obj.env_subsystem_kwargs[i].pop('env_method')
            env_kwargs = in_obj.env_subsystem_kwargs[i]
            env_kwargs['nproc'] = nproc
            env_kwargs['pmem'] = pmem
            env_kwargs['scr_dir'] = scr_dir
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
    supersystem_kwargs['nproc'] = nproc
    supersystem_kwargs['pmem'] = pmem
    supersystem_kwargs['scr_dir'] = scr_dir
    supersystem = cluster_supersystem.ClusterSuperSystem(subsystems, 
        ct_method, **supersystem_kwargs)
    supersystem.freeze_and_thaw()
    #supersystem.env_in_env_energy()
    supersystem.get_active_energy()
    super_energy = supersystem.get_supersystem_energy()

    total_energy = super_energy - supersystem.subsystems[0].env_energy + supersystem.subsystems[0].active_energy

    print("".center(80, '*'))
    print(f"Total Embedding Energy:     {total_energy}")
    print("".center(80,'*'))

    return total_energy

    
class arguments():

    def __init__(self):
        self.ppn       = None
        self.pmem      = None
        self.inp_file  = None
        self.scr_dir   = None
        self.get_args()

    def get_args(self):
 
        parser = ArgumentParser(description='Read Input File')

        parser.add_argument('input_file', nargs=1, default=sys.stdin,
                            help='The input file to submit.')
        parser.add_argument('-x', '--scr', help='Scratch folder to use.',
                            type=str)
        parser.add_argument('-p', '--ppn', help='The processors per node to use.',
                            type=int)
        parser.add_argument('-m', '--pmem', help='Request a particular amount of '
                            'memory per processor. Given in MB.', type=float)

        args = parser.parse_args()

        self.ppn = args.ppn
        self.pmem = args.pmem
        self.inp_file = args.input_file[0]
        self.scf_dir = args.scr



if __name__=='__main__':
    main()

