#!/usr/bin/env python3

# A module to run an input file and get final embedding results ouput to stdout
# Daniel Graham
# Dhabih V. Chulhai

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from textwrap import dedent
import sys
import os
from os.path import expanduser, expandvars, abspath, splitext

import inp_reader, cluster_subsystem, interaction_mediator
import numpy as np
from copy import deepcopy as copy
#import periodic_subsystem


def main():

    IntroText()

    run_args = arguments()
    file_path = abspath(expandvars(expanduser(run_args.inp_file)))
    nproc = run_args.ppn
    pmem = run_args.pmem
    scr_dir = run_args.scr_dir

    in_obj = inp_reader.InpReader(file_path)
    subsystems = []
    for i in range(len(in_obj.subsys_mols)):
        mol = in_obj.subsys_mols[i]
        env_method = in_obj.env_subsystem_kwargs[i].pop('env_method')
        env_kwargs = in_obj.env_subsystem_kwargs[i]
        if not "nproc" in env_kwargs.keys():
            env_kwargs['nproc'] = nproc
        if not "pmem" in env_kwargs.keys():
            env_kwargs['pmem'] = pmem
        if not "scr_dir" in env_kwargs.keys():
            env_kwargs['scrdir'] = scr_dir
        if in_obj.hl_subsystem_kwargs[i] is not None:
            hl_method = in_obj.hl_subsystem_kwargs[i].pop('hl_method')
            hl_kwargs = in_obj.hl_subsystem_kwargs[i]
            hl_kwargs.update(env_kwargs)
            subsys = cluster_subsystem.ClusterHLSubSystem(mol, env_method, hl_method,  **hl_kwargs)
        else:
            subsys = cluster_subsystem.ClusterEnvSubSystem(mol, env_method, **env_kwargs)

        subsystems.append(subsys)

    int_med = interaction_mediator.InteractionMediator(subsystems, supersystem_kwargs=in_obj.supersystem_kwargs, filename=in_obj.inp.filename)
    int_med.do_embedding()
    total_energy = int_med.get_emb_energy()

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
        self.scr_dir = args.scr


def IntroText():

    print (''.center(80, '*'))
    big_title = (" ____  _____      ___  ___ _____ \n"
                 "|  _ \/  ___|     |  \/  ||  ___|\n"
                 "| | | \ `--.  ___ | .  . || |__  \n"
                 "| | | |`--. \/ _ \| |\/| ||  __| \n"
                 "\ \/' /\__/ / (_) | |  | || |___ \n"
                 " \_/\_\____/ \___/\_|  |_/\____/ ")
    for s in big_title.split('\n'):
        print (s.center(80))
    print ('')
    print ('Quantum Solid state and Molecular Embedding'.center(80))
    print ('Version 0.6.0'.center(80))
    print ('github.com/Goodpaster/QSoME'.center(80))
    print ('DOI: 10.5281/zenodo.3356913'.center(80))
    print ('Program Citation: QSoME 0.5, Graham, D.S.; Chulhai, D. V.; Wen, X.; Goodpaster, J. D. University of Minnesota, Minneapolis MN, 2019.'.center(80))
    print ('')

    print ('Authors'.center(80))
    print ('-------'.center(80))
    print ('Daniel S. Graham (graha682@umn.edu)'.center(80))
    print ('Dhabih V. Chulhai (chulhaid@uindy.edu)'.center(80))
    print ('Jason D. Goodpaster (jgoodpas@umn.edu)'.center(80))
    print ('')

    print ('Citations'.center(80))
    print ('---------'.center(80))
    print ('1. J. Chem. Theory Comput. 2017, 13, 1503--1508.'.ljust(50).center(80))
    print ('2. J. Chem. Theory Comput. 2018, 14, 1928--1942.'.ljust(50).center(80))
    print ('')


if __name__=='__main__':
    main()

