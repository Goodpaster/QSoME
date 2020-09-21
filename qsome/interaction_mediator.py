#!/usr/bin/env python
"""An interaction mediator object to handle subsystem object interaction.
Daniel S. Graham
"""

from qsome import helpers
from qsome import cluster_subsystem, cluster_supersystem

def combine_subsystems(subsystems, env_method, fs_kwargs=None):
    """Combines multiple subsystems into one larger subsystem object.

    Parameters
    __________
    subsystems : array
        A list of subsystem objects to combine
    env_method : str
        The name of the environment method to use for the combined subsystem
    fs_kwargs : dict
        Settings for the full system. Determines if subsystem is unrestricted
        (default is None)

    Returns
    -------
    ClusterEnvSubSystem
        The combined subsystem object.
    """
    mol_list = [x.mol for x in subsystems]
    mol = helpers.concat_mols(mol_list)
    sub_order = subsystems[0].env_order
    sub_unrestricted = False
    if fs_kwargs is not None:
        if 'fs_unrestricted' in fs_kwargs.keys():
            sub_unrestricted = fs_kwargs['fs_unrestricted']
    #Need a way of specifying the settings of the implied subsystem somehow.
    return cluster_subsystem.ClusterEnvSubSystem(mol, env_method,
                                                 env_order=sub_order,
                                                 unrestricted=sub_unrestricted)

def gen_supersystems(sup_kwargs, subsystems, filename, scrdir):
    """Generates the list of supersystems made from all subsystems.

    Parameters
    __________
    sup_kwargs : array
        An array of supersystem kwarg dictionaries.
    subsystems : array
        An array of subsystem objects.
    filename : str
        The name of the input file.
    scrdir : str
        The path to the scratch directory

    Returns
    _______
    Array
        A list of supersystem objects.
    """
    supersystems = []
    sorted_subs = sorted(subsystems, key=lambda x: x.env_order)
    while len(sorted_subs) > 0:
        curr_order = sorted_subs[0].env_order
        curr_method = sorted_subs[0].env_method
        curr_sup_kwargs = {}
        sub_sup_kwargs = {}
        if sup_kwargs is not None:
            match_sup_kwargs = [x for x in sup_kwargs if x['env_order'] == curr_order]
            sub_sup_kwargs = [x for x in sup_kwargs if x['env_order'] == (curr_order + 1)]
            assert len(match_sup_kwargs) < 2, 'Ambigious supersystem settings'
            curr_sup_kwargs = match_sup_kwargs[0]
            curr_sup_kwargs.pop('fs_method', None)
        higher_order_subs = [x for x in sorted_subs if x.env_order > curr_order]
        sub_list = []
        while len(sorted_subs) > 0 and sorted_subs[0].env_order == curr_order:
            sub_list.append(sorted_subs.pop(0))
        if len(higher_order_subs) > 0:
            if len(sub_sup_kwargs) > 0:
                combined_subs = combine_subsystems(higher_order_subs,
                                                   curr_method,
                                                   fs_kwargs=sub_sup_kwargs[0])
            else:
                combined_subs = combine_subsystems(higher_order_subs, curr_method)

            sub_list.append(combined_subs)
        curr_sup_kwargs['env_order'] = curr_order
        curr_sup_kwargs['filename'] = filename
        curr_sup_kwargs['scr_dir'] = scrdir
        supersystem = cluster_supersystem.ClusterSuperSystem(sub_list,
                                                             curr_method,
                                                             **curr_sup_kwargs)
        supersystems.append(supersystem)

    return supersystems

class InteractionMediator:
    """
    Interaction Mediator design object for embedding.

    Takes a list of subsystems and a list of keyword arguments to define
    the subsequent supersystems created from thsoe subsystems. Then allows
    various operations to be performed on the embedding system as a whole.

    TODO
    ____
    Add options for periodic settings
    Add option for analytical nuclear gradients

    Parameters
    ----------
    subsystems : array
        Array of subsystem objects dividing the full embedding system
    supersystem_kwargs : array
        Array of dictionaries containing args for init supersystem objects
        (default is None)
    filename : str
        Name of input file
        (default is None)
    nproc : int
        Number of processor cores used for the calculation
        (default is None)
    pmem : float
        Amount of memory per core (in MB)
        (default is None)
    scrdir : str
        Full path to the scratch directory for the calculation
        (default is None)

    Attributes
    ----------
    subsystems : array
        Array of subsystem objects dividing the full embedding system
    nproc : int
        Number of processor cores used for the calculation
    pmem : float
        Amount of memory per core (in MB)
    supersystems : array
        Array of supersystem objects which are combinations of subsystems
    """

    def __init__(self, subsystems, supersystem_kwargs=None, filename=None,
                 nproc=None, pmem=None, scrdir=None):

        self.subsystems = subsystems
        self.nproc = nproc
        self.pmem = pmem
        self.supersystems = gen_supersystems(supersystem_kwargs,
                                             subsystems, filename,
                                             scrdir)
        self.set_chkfile_index()
        self.init_density()

    def set_chkfile_index(self, index=0):
        """Sets the index for each supersystem in the checkpoint file.

        Parameters
        ----------
        index : int
            Starting index for the checkpoint file
            (default is 0)
        """

        self.chkfile_index = index
        for i in range(len(self.supersystems)):
            sup = self.supersystems[i]
            sup.set_chkfile_index(i)

    def init_density(self):
        """Generates the initial density guess for each subsystem"""

        for sup in self.supersystems:
            sup.init_density()

    def do_embedding(self):
        """Perform the embedding calculation for the full system.

        This function iterates through all supersystem objects and relaxes
        the electron density for each within the embedding framework.
        Potential energy surfaces are passed to ensure full relaxation.
        Once low level calculations are complete, the high level calculation
        is performed within the low level potential of the system.
        """

        ext_potential = [0., 0.]
        for i in range(len(self.supersystems)):
            curr_sup = self.supersystems[i]
            curr_sup.ext_pot = ext_potential
            curr_sup.freeze_and_thaw()
            #There is an issue here.
            #new_ext_pot = curr_sup.get_emb_ext_pot() #This method formats the external potential for the next super system
            #ext_potential = new_ext_pot

    def get_emb_energy(self):
        """Prints the embedding energy.

        Iterates through all supersystem and prints the energy of each
        component. Then prints the energy of the entire system to give
        the final embedding energy.
        """

        #Add all the components together to get an energy summary.
        energy_tot = 0
        for i in range(len(self.supersystems) - 1):
            sup = self.supersystems[i]
            sup_e = sup.get_supersystem_energy()
            sup.get_env_energy()
            sub_e = sup.subsystems[-1].env_energy
            print(f"Supersystem {i + 1} Energy: {sup_e}")
            print(f"Higher level subsystem Energy: {sub_e}")
            energy_tot += sup_e - sub_e

        sup = self.supersystems[-1]
        sup_e = sup.get_supersystem_energy()
        env_in_env_e = sup.get_env_in_env_energy()
        sup.get_hl_energy()
        sup.get_env_energy()
        energy_tot += sup_e
        for sub in sup.subsystems:
            #THIS IS VERY HACKY. NEED TO COME UP WITH A BETTER WAY TO GET SUBSYSTEM STUFF.
            if 'hl_energy' in vars(sub):
                energy_tot -= sub.env_energy
                energy_tot += sub.hl_energy

        print("".center(80, '*'))
        print(f"Total Embedding Energy:     {energy_tot}")
        print("".center(80, '*'))
        return energy_tot
