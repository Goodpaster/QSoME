#!/usr/bin/env python
"""A module to define the input reader object
Daniel S. Graham
Dhabih V. Chulhai
"""

from __future__ import print_function, division

from copy import copy
import input_reader
import numpy as np
from pyscf import gto, pbc
from qsome import helpers

class InputError(Exception):
    """Exception for an improperly designed input file.

    Some input options conflict with each other or are nonsensical.
    While these will not raise errors with the input reader module,
    they should raise an error for embedding.

    Parameters
    ----------
    message : str
        Human readable message describing details of the error.

    Attributes
    ----------
    message : str
        Human readable message describing details of the error.
    """


    def __init__(self, message):
        super().__init__()
        self.message = message

def read_input(filename):
    """Reads a formatted input file, generates an InputReader object.

    Parameters
    ---------
    filename : str
        Path to input file.
    """

    reader = input_reader.InputReader(comment=['!', '#', '::', '//'],
                                      case=False, ignoreunknown=False)
    subsys = reader.add_block_key('subsystem', required=True, repeat=True)
    add_subsys_settings(subsys)

    # Define the environment settings and embedding operations.
    env_settings = reader.add_block_key('env_method_settings',
                                        required=True,
                                        repeat=True)
    env_settings.add_line_key('env_order', type=int)
    env_settings.add_line_key('env_method', type=str, required=True)
    env_settings.add_line_key('grad', type=float)
    env_settings.add_line_key('cycles', type=int)
    env_settings.add_line_key('grid', type=int)
    env_settings.add_line_key('rhocutoff', type=float)
    env_settings.add_line_key('verbose', type=int)
    env_settings.add_boolean_key('compare_density')
    excited_settings = env_settings.add_block_key('excited_settings')
    add_excited_settings(excited_settings)
    add_env_settings(env_settings)

    # Freeze and thaw settings
    embed = env_settings.add_block_key('embed_settings')
    excited_settings = embed.add_block_key('excited_settings')
    add_excited_settings(excited_settings)
    add_embed_settings(embed)

    # Define the high level calculation settings.
    hl_settings = reader.add_block_key('hl_method_settings', repeat=True,
                                       required=True)
    hl_settings.add_line_key('hl_order', type=int)
    hl_settings.add_line_key('hl_method', type=str)
    add_hl_settings(hl_settings)


    cc_settings = hl_settings.add_block_key('cc_settings')
    add_cc_settings(cc_settings)

    cas_settings = hl_settings.add_block_key('cas_settings')
    add_cas_settings(cas_settings)

    shci_settings = hl_settings.add_block_key('shci_settings')
    add_shci_settings(shci_settings)

    dmrg_settings = hl_settings.add_block_key('dmrg_settings')
    dmrg_settings.add_line_key('maxM', type=int)
    dmrg_settings.add_line_key('num_thirds', type=int)

    excited_settings = hl_settings.add_block_key('excited_settings')
    add_excited_settings(excited_settings)

    basis = reader.add_block_key('basis')
    basis.add_regex_line('basis_def', r'\s*([A-Za-z.:]+[.:\-]?\d*)\s+.+',
                         repeat=True)
    ecp = reader.add_block_key('ecp')
    ecp.add_regex_line('ecp_def', r'\s*([A-Za-z.:]+[.:\-]?\d*)\s+.+',
                       repeat=True)
    reader.add_line_key('unit', type=('angstrom', 'a', 'bohr', 'b'))

    reader.add_line_key('ppmem', type=(int, float)) # MB
    reader.add_line_key('nproc', type=int)
    reader.add_line_key('scrdir', type=str)
    inp = reader.read_input(filename)
    inp.filename = filename

    print("".center(80, '*'))
    print("Input File".center(80))
    print("".center(80, '*'))
    with open(filename, 'r') as fout:
        print(fout.read())
    print("".center(80, '*'))
    return inp

def add_subsys_settings(subsys_block):
    """Adds the subsystem settings to the block.

    Parameters
    ----------
    subsys_block : input_reader block object
        The subsystem block which to add options.
    """

    # Cannot use repeating patterns to shorten: input_reader uses re.
    subsys_block.add_regex_line(
        'atoms',
        (r'\s*([A-Za-z.:]+[.:\-]?\d*)'
         r'\s+(\-?\d+\.?\d*)\s+(\-?\d+\.?\d*)\s+(\-?\d+\.?\d*)'),
        repeat=True)
    subsys_block.add_line_key('charge', type=int)
    subsys_block.add_line_key('spin', type=int)
    subsys_block.add_line_key('unit', type=('angstrom', 'a', 'bohr', 'b'))
    subsys_block.add_line_key('diis', type=int)
    subsys_block.add_boolean_key('addlinkbasis') # Add link ghost atoms
    sub_basis = subsys_block.add_block_key('basis')
    sub_basis.add_regex_line('basis_def',
                             r'\s*([A-Za-z.:]+[.:\-]?\d*)\s+.+',
                             repeat=True)
    sub_ecp = subsys_block.add_block_key('ecp')
    sub_ecp.add_regex_line('ecp_def', r'\s*([A-Za-z.:]+[.:\-]?\d*)\s+.+',
                           repeat=True)
    subsys_block.add_line_key('env_method_num', type=int)
    subsys_block.add_line_key('hl_method_num', type=int)

    #Override default environment method settings.
    sub_env_settings = subsys_block.add_block_key('env_method_settings')
    sub_env_settings.add_line_key('subcycles', type=int)
    sub_env_settings.add_line_key('setfermi', type=float)
    sub_env_settings.add_boolean_key('freeze')
    add_env_settings(sub_env_settings)

    excited_settings = sub_env_settings.add_block_key('excited_settings')
    add_excited_settings(excited_settings)

    # Override default high level method settings.
    sub_hl_settings = subsys_block.add_block_key('hl_method_settings')
    add_hl_settings(sub_hl_settings)

    sub_cc_settings = sub_hl_settings.add_block_key('cc_settings')
    add_cc_settings(sub_cc_settings)

    sub_cas_settings = sub_hl_settings.add_block_key('cas_settings')
    add_cas_settings(sub_cas_settings)

    sub_shci_settings = sub_hl_settings.add_block_key('shci_settings')
    add_shci_settings(sub_shci_settings)

    sub_dmrg_settings = sub_hl_settings.add_block_key('dmrg_settings')
    sub_dmrg_settings.add_line_key('maxM', type=int)
    sub_dmrg_settings.add_line_key('num_thirds', type=int)

    excited_settings = sub_hl_settings.add_block_key('excited_settings')
    add_excited_settings(excited_settings)

def add_embed_settings(embed_block):
    """Adds the embedding settings to the input reader block.

    Parameters
    ----------
    embed_block : input_reader block object
        The input block to which to add embedding setting options.
    """

    embed_block.add_line_key('cycles', type=int)
    embed_block.add_line_key('subcycles', type=int)
    embed_block.add_line_key('basis_tau', type=float)
    embed_block.add_line_key('conv', type=float)
    embed_block.add_line_key('grad', type=float)
    embed_block.add_line_key('damp', type=float)
    embed_block.add_line_key('diis', type=int) # Use DIIS for fock. (0 for off)
    embed_block.add_line_key('setfermi', type=float)

    # 0 is after F&T cycle, otherwise after every n subsystem cycles
    embed_block.add_line_key('updatefock', type=int)
    embed_block.add_line_key('updateproj', type=int)

    # Initial guess for the subsystem embedding calculation
    embed_block.add_line_key('initguess', type=(
        'minao', 'atom', '1e', 'readchk', 'supmol', 'submol', 'localsup'))
    embed_block.add_boolean_key('unrestricted')

    # Output subsystem orbitals after F&T cycles
    embed_block.add_boolean_key('save_orbs')
    embed_block.add_boolean_key('save_density')
    embed_block.add_boolean_key('save_spin_density')

    # This section needs work. Should be uniform option for setting op.
    operator = embed_block.add_mutually_exclusive_group(dest='proj_oper',
                                                        required=False)
    operator.add_line_key('mu', type=float, default=1e6)
    operator.add_boolean_key('manby', action=1e6)
    operator.add_boolean_key('huzinaga', action='huz')
    # Fermi shifted
    operator.add_boolean_key('huzinagafermi', action='huzfermi')
    operator.add_boolean_key('huzfermi', action='huzfermi')

    embed_block.add_boolean_key('excited_relax')

def add_env_settings(inp_block):
    """Adds the environment subsystem settings to the block.

    Parameters
    ----------
    inp_block : input_reader block object
        The input block to add environment setting options to.
    """

    inp_block.add_line_key('initguess', type=('minao', 'atom', '1e', 'readchk',
                                              'supmol', 'submol'))
    inp_block.add_line_key('smearsigma', type=float)
    inp_block.add_line_key('conv', type=float)
    inp_block.add_line_key('damp', type=float)
    inp_block.add_line_key('shift', type=float)
    inp_block.add_line_key('diis', type=int)
    inp_block.add_boolean_key('unrestricted')
    inp_block.add_boolean_key('density_fitting')
    inp_block.add_boolean_key('save_orbs')
    inp_block.add_boolean_key('save_density')
    inp_block.add_boolean_key('save_spin_density')
    inp_block.add_boolean_key('excited')

def add_hl_settings(inp_block):
    """Adds the high level subsystem settings to the block.

    Parameters
    ----------
    inp_block : input_reader block object
        The input block to add high level setting options to.
    """

    inp_block.add_line_key('initguess', type=('minao', 'atom', '1e', 'readchk',
                                              'supmol', 'submol'))
    inp_block.add_line_key('spin', type=int)
    inp_block.add_line_key('conv', type=float)
    inp_block.add_line_key('grad', type=float)
    inp_block.add_line_key('cycles', type=int)
    inp_block.add_line_key('damp', type=float)
    inp_block.add_line_key('shift', type=float)
    inp_block.add_line_key('use_ext', type=('molpro', 'bagel', 'molcas',
                                            'openmolcas'))
    inp_block.add_boolean_key('compress_approx')
    inp_block.add_boolean_key('unrestricted')
    inp_block.add_boolean_key('density_fitting')
    inp_block.add_boolean_key('save_orbs')
    inp_block.add_boolean_key('save_density')
    inp_block.add_boolean_key('save_spin_density')
    inp_block.add_boolean_key('excited')

def add_cc_settings(inp_block):
    """Adds the high level CC settings to the block.

    Parameters
    ----------
    inp_block : input_reader block object
        The input block to add high level CC setting options to.
    """

    inp_block.add_boolean_key('loc_orbs')
    inp_block.add_line_key('cc_initguess', type=str)
    inp_block.add_line_key('froz_core_orbs', type=str)

def add_cas_settings(inp_block):
    """Adds the high level CAS settings to the block.

    Parameters
    ----------
    inp_block : input_reader block object
        The input block to add high level CAS setting options to.
    """

    inp_block.add_boolean_key('loc_orbs')
    inp_block.add_line_key('cas_initguess', type=str)
    inp_block.add_line_key('active_orbs', type=str)
    inp_block.add_line_key('avas', type=str)

def add_shci_settings(inp_block):
    """Adds the high level SHCI settings to the block.

    Parameters
    ----------
    inp_block : input_reader block object
        The input block to add high level SHCI setting options to.
    """

    inp_block.add_line_key('mpi_prefix', type=str)
    inp_block.add_line_key('sweep_iter', type=str)
    inp_block.add_line_key('sweep_epsilon', type=str)
    inp_block.add_line_key('nPTiter', type=int)
    inp_block.add_boolean_key('no_stochastic')
    inp_block.add_boolean_key('NoRDM')

def add_excited_settings(inp_block):
    """Adds the excited state settings to the block.

    Parameters
    ----------
    inp_block : input_reader block object
        The input block to add excited state setting options.
    """

    inp_block.add_line_key('nroots', type=int, default=3)
    inp_block.add_line_key('conv', type=float)
    inp_block.add_line_key('cycles', type=int)
    inp_block.add_line_key('eom_type', type=('ee', 'ea', 'ip', str),
                        default='ee')
    inp_block.add_boolean_key('koopmans')
    # IP/EA-EOM-CCSD(T)*a by Matthews and Stanton
    # https://github.com/pyscf/pyscf-doc/blob/master/examples/pbc/29-eom_ccsd_Ta.py
    # https://aip.scitation.org/doi/10.1063/1.4962910
    inp_block.add_boolean_key('Ta_star')

def cleanup_keys(settings_dict, key_correct=None):
    """Removes unnessecary keys created by input_reader.

    Parameters
    ----------
    settings_dict : dict
        A dictionary of the system settings to be formatted
    key_correct : dict
        A dictionary of strings to modify the inp object keys to be kwargs.
    """

    list_keys = list(settings_dict.keys())
    for k in list_keys:
        if k.startswith('_'):
            settings_dict.pop(k)
        elif settings_dict[k] is None:
            settings_dict.pop(k)

    if key_correct:
        list_keys = list(settings_dict.keys())
        for correct in key_correct:
            if correct in list_keys:
                settings_dict[key_correct[correct]] = settings_dict.pop(correct)

def get_universal_keys(inp_dict):
    """Creates the kwarg dictionary for systemwide parameters.

    Parameters
    ----------
    inp_dict : dict
        A dictionary of kwargs generated using input_reader.
    """

    uni_subsys_settings = {}
    universal_settings_keys = ['filename', 'ppmem', 'nproc', 'scrdir']
    for universal_key in universal_settings_keys:
        if inp_dict[universal_key]:
            uni_subsys_settings[universal_key] = inp_dict[universal_key]
    return uni_subsys_settings

def build_hl_dict(hl_settings, hl_params):
    """Builds the high level kwarg dictionary from the input reader dict.

    Parameters
    ----------
    hl_settings : dict
        Dictionary of hl subsystem kwargs.
    hl_params : dict
        Dictionary of input_reader hl settings.
    """

    base_setting_kwargs = ['hl_method', 'initguess', 'spin', 'conv',
                           'grad', 'cycles', 'damp', 'shift',
                           'compress_approx', 'unrestricted',
                           'density_fitting', 'save_orbs', 'save_density',
                           'save_spin_density', 'use_ext', 'excited']

    for base_kwarg in base_setting_kwargs:
        if base_kwarg in hl_params and hl_params[base_kwarg]:
            hl_settings[base_kwarg] = hl_params[base_kwarg]
    if hl_params['cc_settings']:
        cc_dict = vars(hl_params['cc_settings'])
        cleanup_keys(cc_dict)
        hl_settings['hl_dict'] = copy(cc_dict)
        if 'froz_core_orbs' in cc_dict:
            froz_split = cc_dict['froz_core_orbs'].split(',')
            froz_list = [int(x) for x in froz_split]
            if len(froz_list) == 1:
                froz_list = froz_list[0]
            hl_settings['hl_dict']['froz_core_orbs'] = froz_list
    if hl_params['cas_settings']:
        cas_dict = vars(hl_params['cas_settings'])
        cleanup_keys(cas_dict)
        hl_settings['hl_dict'] = copy(cas_dict)
        if 'active_orbs' in cas_dict:
            act_split = cas_dict['active_orbs'].split(',')
            act_list = [int(x) for x in act_split]
            hl_settings['hl_dict']['active_orbs'] = act_list
        if 'avas' in cas_dict:
            avas_split = cas_dict['avas'].split(',')
            avas_list = [str(x) for x in avas_split]
            hl_settings['hl_dict']['avas'] = avas_list
    if hl_params['shci_settings']:
        shci_dict = vars(hl_params['shci_settings'])
        cleanup_keys(shci_dict)
        hl_settings['hl_dict'] = shci_dict
    if hl_params['dmrg_settings']:
        dmrg_dict = vars(hl_params['dmrg_settings'])
        cleanup_keys(dmrg_dict)
        hl_settings['hl_dict'] = dmrg_dict
    if hl_params['excited_settings']:
        excited_dict = vars(hl_params['excited_settings'])
        cleanup_keys(excited_dict)
        hl_settings['hl_excited_dict'] = excited_dict

def add_ghost_link(subsys_mols, sub_settings):
    """Adds linking ghost atoms to the subsystem mol objects.

    Parameters
    ----------
    subsys_mols : list
        A list of mol objects for the different subsystems.
    sub_settings : list
        A list of subsystem settings
    """

    max_dist = 3.8
    ghost_basis = '3-21g'

    #First get array of interatom distances
    coord_array = np.asarray(np.vstack([x.atom_coords() for x in subsys_mols]))
    inter_dist = gto.inter_distance(None, coords=coord_array)
    close_indices = np.argwhere(inter_dist <= max_dist)
    #Iterate through all close indices and add link atoms to the sub on one conditional line.
    lowest_index = 0
    for i, subsystem in enumerate(sub_settings):
        num_atoms = len(subsys_mols[i].atom)
        high_index = lowest_index + num_atoms
        if subsystem.addlinkbasis:
            ghost_mol = gto.M()
            ghost_mol.atom = []
            ghost_mol.basis = {}
            for index in close_indices:
                if ((index[0] >= lowest_index and index[0] < high_index) and
                        (index[1] < lowest_index or index[1] >= high_index)):
                    atm1_coord = coord_array[index[0]]
                    atm2_coord = coord_array[index[1]]
                    if subsystem.basis:
                        new_atom, new_basis = helpers.gen_link_basis(atm1_coord,
                                                                     atm2_coord,
                                                                     subsystem.basis)
                    else:
                        new_atom, new_basis = helpers.gen_link_basis(atm1_coord,
                                                                     atm2_coord,
                                                                     ghost_basis)

                    ghost_mol.atom.append(new_atom)
                    ghost_mol.basis.update(new_basis)

            ghost_mol.build(unit='bohr')
            subsys_mols[i] = gto.conc_mol(subsys_mols[i], ghost_mol)
        lowest_index = high_index

class InpReader:
    """
    Reads a specific form of input file for creating objects.

    Takes a formatted text file and generates arg and kwarg
    dictionaries to generate the specified embedding system objects.

    Parameters
    ----------
    filename : str
        Path to input file


    Attributes
    ----------
    inp : InputReader
        The InputReader object of the input file
    env_subsystem_kwargs : dict
        Dictionary of settings for creating ClusterEnvSubSystem object
    hl_settings_kwargs : dict
        Dictionary of settings for creating ClusterHLSubSystem object
    supersystem_kwargs : dict
        Dictionary of settings for creating ClusterSupersystem object
    subsys_mols : list
        A list of subsystem pyscf Mol objects
    """


    def __init__(self, filename):

        self.inp = read_input(filename)
        self.env_subsystem_kwargs = self.get_env_subsystem_kwargs()
        self.hl_subsystem_kwargs = self.get_hl_subsystem_kwargs()
        self.supersystem_kwargs = self.get_supersystem_kwargs()
        self.subsys_mols = self.gen_mols()


    def get_supersystem_kwargs(self, inp=None):
        """Generates a kwarg dictionary for supersystem object.

        Parameters
        ----------
        inp : InputReader, optional
            InputReader object to extract supersystem settings from.
            (default is None)
        """

        if inp is None:
            inp = self.inp

        # universal settings
        inp_dict = vars(inp)
        universal_settings = get_universal_keys(inp_dict)

        supersystem_kwargs = []
        # Setup supersystem method

        fs_dict = {'env_order'            :       'env_order',
                   'env_method'           :       'fs_method',
                   'smearsigma'           :       'fs_smearsigma',
                   'initguess'            :       'fs_initguess',
                   'conv'                 :       'fs_conv',
                   'grad'                 :       'fs_grad',
                   'cycles'               :       'fs_cycles',
                   'damp'                 :       'fs_damp',
                   'shift'                :       'fs_shift',
                   'diis'                 :       'fs_diis',
                   'grid'                 :       'fs_grid_level',
                   'rhocutoff'            :       'fs_rhocutoff',
                   'verbose'              :       'fs_verbose',
                   'unrestricted'         :       'fs_unrestricted',
                   'density_fitting'      :       'fs_density_fitting',
                   'compare_density'      :       'compare_density',
                   'save_orbs'            :       'fs_save_orbs',
                   'save_density'         :       'fs_save_density',
                   'save_spin_density'    :       'fs_save_spin_density',
                   'excited'              :       'fs_excited',
                   'excited_settings'     :       'fs_excited_dict'}

        ft_dict = {'cycles'               :       'ft_cycles',
                   'subcycles'            :       'ft_subcycles',
                   'basis_tau'            :       'ft_basis_tau',
                   'conv'                 :       'ft_conv',
                   'grad'                 :       'ft_grad',
                   'damp'                 :       'ft_damp',
                   'diis'                 :       'ft_diis',
                   'setfermi'             :       'ft_setfermi',
                   'updatefock'           :       'ft_updatefock',
                   'initguess'            :       'ft_initguess',
                   'unrestricted'         :       'ft_unrestricted',
                   'save_orbs'            :       'ft_save_orbs',
                   'save_density'         :       'ft_save_density',
                   'save_spin_density'    :       'ft_save_spin_density',
                   'proj_oper'            :       'ft_proj_oper',
                   'excited_relax'        :       'ft_excited_relax',
                   'excited_settings'     :       'ft_excited_dict'}

        for supersystem in inp.env_method_settings:
            sup_settings_dict = vars(supersystem)
            sup_kwargs = {}
            for set_key in fs_dict:
                if sup_settings_dict[set_key] is not None:
                    if set_key == 'excited_settings':
                        sup_settings_dict[set_key] = vars(sup_settings_dict[set_key])
                        cleanup_keys(sup_settings_dict[set_key])
                    sup_kwargs[fs_dict[set_key]] = sup_settings_dict[set_key]

            if sup_settings_dict['embed_settings'] is not None:
                emb_set_dict = vars(sup_settings_dict['embed_settings'])
                if emb_set_dict['excited_settings']:
                    excited_dict = vars(emb_set_dict['excited_settings'])
                    cleanup_keys(excited_dict)
                    emb_set_dict['excited_settings'] = excited_dict
                for s_key in ft_dict:
                    if emb_set_dict[s_key] is not None:
                        sup_kwargs[ft_dict[s_key]] = emb_set_dict[s_key]

            sup_kwargs.update((universal_settings).copy())
            supersystem_kwargs.append(sup_kwargs)


        return supersystem_kwargs

    def get_env_subsystem_kwargs(self, inp=None):
        """Generates a kwarg dictionary for ClusterEnvSubSystem object.

        Parameters
        ----------
        inp : InputReader, optional
            InputReader object to extract subsystem settings from.
            (default is None)
        """

        if inp is None:
            inp = self.inp

        # subsystem universal settings
        inp_dict = vars(inp)
        uni_subsys_settings = get_universal_keys(inp_dict)

        env_kwargs = []
        env_dict = {'smearsigma'    : 'env_smearsigma'}
        for subsystem in inp.subsystem:

            env_settings = {}

            if subsystem.env_method_num is None:
                if len(inp.env_method_settings) > 1:
                    #Throw error. Unclear which env method to use
                    raise InputError(('Multiple environment methods available.'
                                      'Either specify which to use in the '
                                      'subsystem or specify only one '
                                      'environment method'))
                env_settings['env_order'] = 1
                setattr(inp.env_method_settings[0], 'env_order', 1)
            else:
                env_settings['env_order'] = subsystem.env_method_num
            if subsystem.diis:
                env_settings['diis'] = subsystem.diis

            params_found = False
            method_num = env_settings['env_order']
            for env_param in inp.env_method_settings:
                if env_param.env_order == method_num and not params_found:
                    params_found = True
                    env_settings['env_method'] = env_param.env_method
                    if env_param.embed_settings:
                        sub_num = env_param.embed_settings.subcycles
                        env_settings['subcycles'] = sub_num
                        if env_param.embed_settings.unrestricted is not None:
                            unres = env_param.embed_settings.unrestricted
                            env_settings['unrestricted'] = unres
                elif env_param.env_order == method_num and params_found:
                    #Ambigious environment parameter specification
                    raise InputError(('Multiple environment methods with the '
                                      'same order number. Each environment '
                                      'method should have a unique identifying'
                                      ' number.'))
            if not params_found:
                #Parameters with the given number not found
                raise InputError(('Environment method not found with number '
                                  'specified in the subsystem.'))

            if subsystem.env_method_settings is not None:
                env_settings.update(vars(subsystem.env_method_settings))

            env_settings.update(uni_subsys_settings.copy())

            #Remove keys put there by input_reader
            cleanup_keys(env_settings, env_dict)
            env_kwargs.append(env_settings)

        return env_kwargs


    def get_hl_subsystem_kwargs(self, inp=None):
        """Generates a kwarg dictionary for ClusterActiveSubSystem object.

        Parameters
        ----------
        inp : InputReader, optional
            InputReader object to extract active subsystem settings from.
            (default is None)
        """

        if inp is None:
            inp = self.inp

        hl_kwargs = []

        hl_dict = {'initguess': 'hl_initguess',
                   'spin' : 'hl_spin',
                   'conv' : 'hl_conv',
                   'grad' : 'hl_grad',
                   'cycles' : 'hl_cycles',
                   'damp' : 'hl_damp',
                   'shift' : 'hl_shift',
                   'compress_approx': 'hl_compress_approx',
                   'unrestricted': 'hl_unrestricted',
                   'save_orbs': 'hl_save_orbs',
                   'save_density': 'hl_save_density',
                   'save_spin_density': 'hl_save_spin_density',
                   'density_fitting': 'hl_density_fitting',
                   'use_ext': 'hl_ext',
                   'excited': 'hl_excited'
                   }


        for subsystem in inp.subsystem:
            if subsystem.hl_method_num is None:
                hl_kwargs.append(None)
            else:
                hl_settings = {'hl_order': subsystem.hl_method_num}
                method_num = subsystem.hl_method_num
                params_found = False
                for hl_param in inp.hl_method_settings:
                    hl_param_dict = vars(hl_param)
                    if (hl_param_dict['hl_order'] == method_num and
                            not params_found):
                        params_found = True
                        build_hl_dict(hl_settings, hl_param_dict)

                    elif (hl_param_dict['hl_order'] == method_num
                          and params_found):
                        #Ambigious environment parameter specification
                        raise InputError(('Multiple high level methods with',
                                          'the same order number. Each hl',
                                          'method should have a unique',
                                          'identifying number.'))
                if not params_found:
                    #Parameters with the given number not found
                    raise InputError(('High level method not found with',
                                      'number specified in the subsystem.'))

                #Update with the subsystem particluar settings
                if subsystem.hl_method_settings is not None:
                    hl_param = subsystem.hl_method_settings
                    hl_param_dict = vars(hl_param)
                    build_hl_dict(hl_settings, hl_param_dict)

                cleanup_keys(hl_settings, hl_dict)
                hl_kwargs.append(hl_settings)

        return hl_kwargs

    def gen_mols(self):
        """Generates the mol or cell objects specified in inp..

        Parameters
        ----------
        inp : InputReader, optional
            InputReader object to extract mol information from.
            (default is None)
        """

        subsys_mols = []
        inp = self.inp
        for subsystem in inp.subsystem:
            atom_list = subsystem.atoms
            mol = gto.Mole()
            mol.atom = []
            for atom in atom_list:
                mol.atom.append([atom.group(1), (float(atom.group(2)),
                                                 float(atom.group(3)),
                                                 float(atom.group(4)))])

            if subsystem.charge:
                mol.charge = subsystem.charge
            if subsystem.spin:
                mol.spin = subsystem.spin
            if subsystem.unit:
                mol.unit = subsystem.unit

            described_basis = {}
            if inp.basis:
                for basis in inp.basis.basis_def:
                    basis_str = basis.group(0)
                    split_basis = basis_str.split()
                    described_basis[split_basis[0]] = split_basis[1]

            if subsystem.basis:
                for basis in subsystem.basis.basis_def:
                    basis_str = basis.group(0)
                    split_basis = basis_str.split()
                    described_basis[split_basis[0]] = split_basis[1]

            if not described_basis:
                print("YOU HAVE NOT DEFINED A BASIS. USING 3-21g BY DEFAULT")
                described_basis['default'] = '3-21g'

            described_ecp = {}
            if inp.ecp is not None:
                for ecp in inp.ecp.ecp_def:
                    ecp_str = ecp.group(0)
                    split_ecp = ecp_str.split()
                    described_ecp[split_ecp[0]] = split_ecp[1]

            if subsystem.ecp is not None:
                for ecp in subsystem.ecp.ecp_def:
                    ecp_str = ecp.group(0)
                    split_ecp = ecp_str.split()
                    described_ecp[split_ecp[0]] = split_ecp[1]

            mol.basis = described_basis
            mol.ecp = described_ecp
            mol.build(dump_input=False)
            subsys_mols.append(mol)

        for sub in subsys_mols:
            sub.build(dump_input=False)

        #Add ghost link atoms Assumes angstroms.
        add_ghost_link(subsys_mols, inp.subsystem)
        return subsys_mols
