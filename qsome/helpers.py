#!/usr/bin/env python
"""A module containing methods used in various places that don't fit in
another module. Maintians good code decoupling
Daniel S. Graham
"""

import copy
import numpy as np
from pyscf import gto

def gen_link_basis(atom1_coord, atom2_coord, basis, basis_atom='H'):
    """Generate the linking ghost atom between two atoms.

    Parameters
    ----------
    atom1 : list
        atom coordinates
    atom2 : list
        atom coordinates
    basis : str
        the basis of the link atom
    basis_atom : str
        The type of atom for the ghost atom.
        (default is 'H')

    Returns
    -------
    tuple
        A tuple of the link atom coordinate and basis
    """

    basis_atom = 'H'
    ghost_name = f'ghost:{basis_atom}'
    midpoint = (atom1_coord + atom2_coord)/2.
    atm = [ghost_name, tuple(midpoint)]
    basis = {ghost_name: gto.basis.load(basis, basis_atom)}
    return (atm, basis)

def __remove_overlap_ghost(mol):
    """Removes overlapping ghost atoms between mol objects

    Parameters
    __________
    mol : Mole object
        The mole object to remove ghost atoms from. Removes them in-place.

    Returns
    -------
    Mole object
        Mole object with overlapping ghost atoms removed
    """

    int_dist = gto.inter_distance(mol)
    same_coords = np.argwhere(int_dist == 0.0)
    remove_list = []
    while len(same_coords) > 0:
        curr_coord = same_coords[-1]
        if curr_coord[0] == curr_coord[1]:
            same_coords = same_coords[:-1]
        else:
            atom1 = mol.atom_symbol(curr_coord[0]).lower()
            atom2 = mol.atom_symbol(curr_coord[1]).lower()
            assert ("ghost" in atom1 or "ghost" in atom2),\
                    f"{atom1.capitalize()} and {atom2.capitalize()} are overlapping!"
            if "ghost" in atom2:
                remove_list.append(curr_coord[1])
            else:
                remove_list.append(curr_coord[0])
            same_coords = same_coords[:-1]
            inverse_coords = (curr_coord[1], curr_coord[0])
            inverse_index = np.argwhere((same_coords == inverse_coords).all(axis=1))
            same_coords = np.delete(same_coords, inverse_index, axis=0)

    for remove_index in sorted(remove_list, reverse=True):
        mol.atom.pop(remove_index)
    mol.build()
    return mol

def concat_mols(mol_list):
    """Concatenates Mole objects into one Mole.

    Parameters
    ----------
    mol_list : list
        List of Mole objects to concatenate into single Mole.
    """

    assert (len(mol_list) > 1), "Must have more than 1 mol object"

    total_spin = 0
    for i, mol_obj in enumerate(mol_list):
        mol1 = gto.mole.copy(mol_obj)
        total_spin += mol1.spin
        mol1.unit = 'bohr'
        atom_coords = mol1.atom_coords()
        mol1.atom = []
        uniq_atoms = set()
        for j, coord in enumerate(atom_coords):
            a_symb = mol1.atom_symbol(j) + '-' + str(i)
            uniq_atoms.add(a_symb)
            mol1.atom.append([a_symb, tuple(coord)])

        if isinstance(mol1.basis, (str, tuple, list)):
            new_basis = dict(((a, mol1.basis) for a in uniq_atoms))
        elif isinstance(mol1.basis, dict):
            old_basis = copy.copy(mol1.basis)
            new_basis = {}
            if 'default' in old_basis:
                default_basis = old_basis['default']
                new_basis = dict(((a, default_basis) for a in uniq_atoms))
                del old_basis['default']
            for atom_symb in old_basis:
                new_symb = atom_symb + '-' + str(i)
                new_basis[new_symb] = old_basis[atom_symb]
        else:
            new_basis = mol1.basis
        mol1.basis = gto.format_basis(new_basis)

        if mol1.ecp:
            if isinstance(mol1.ecp, str):
                new_ecp = dict(((a, str(mol1.ecp)) for a in uniq_atoms))
            elif isinstance(mol1.ecp, dict):
                old_ecp = copy.copy(mol1.ecp)
                if 'default' in old_ecp:
                    default_ecp = old_ecp['default']
                    new_ecp = dict(((a, default_ecp) for a in uniq_atoms))
                    del old_ecp['default']
                for atom_symb in old_ecp:
                    new_symb = atom_symb + '-' + str(i)
                    new_ecp[new_symb] = old_ecp[atom_symb]
            else:
                new_ecp = mol1.ecp
            mol1.ecp = gto.format_ecp(new_ecp)

        mol1.build()
        if i == 0:
            mol2 = gto.mole.copy(mol1)
        else:
            new_mol = gto.mole.conc_mol(mol1, mol2)
            new_mol.build()
            mol2 = new_mol

    conc_mol = mol2
    conc_mol.spin = total_spin #check that this works properly.
    #Remove overlapping ghost atoms.
    final_mol = __remove_overlap_ghost(conc_mol)
    final_mol = conc_mol
    return final_mol
