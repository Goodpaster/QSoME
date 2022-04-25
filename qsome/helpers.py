#!/usr/bin/env python
"""A module containing methods used in various places that don't fit in
another module. Maintians good code decoupling
Daniel S. Graham
"""

import copy
import numpy as np
from pyscf import gto, scf, dft
from pyscf.dft import numint

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
            new_mol = gto.mole.conc_mol(mol2, mol1)
            new_mol.build()
            mol2 = new_mol

    conc_mol = mol2
    conc_mol.spin = total_spin #check that this works properly.
    #Remove overlapping ghost atoms.
    final_mol = __remove_overlap_ghost(conc_mol)
    final_mol = conc_mol
    return final_mol

def gen_scf_obj(mol, scf_method, **kwargs):
    """Generates an scf object setting all relevant parameters for use later in
    embedding"""

    if 'unrestricted' in kwargs and kwargs.get('unrestricted'):
        if 'hf' in scf_method:
            scf_obj = scf.UHF(mol)
        else:
            scf_obj = scf.UKS(mol)
            scf_obj.xc = scf_method
    elif mol.spin != 0:
        if 'hf' in scf_method:
            scf_obj = scf.ROHF(mol)
        else:
            scf_obj = scf.ROKS(mol)
            scf_obj.xc = scf_method
    else:
        if 'hf' in scf_method:
            scf_obj = scf.RHF(mol)
        else:
            scf_obj = scf.RKS(mol)
            scf_obj.xc = scf_method

    if 'diis_num' in kwargs:
        if kwargs.get('diis_num') == 0:
            scf_obj.DIIS = None
        if kwargs.get('diis_num') == 1:
            scf_obj.DIIS = scf.CDIIS
        if kwargs.get('diis_num') == 2:
            scf_obj.DIIS = scf.EDIIS
        if kwargs.get('diis_num') == 3:
            scf_obj.DIIS = scf.ADIIS

    if 'grid_level' in kwargs:
        grids = dft.gen_grid.Grids(mol)
        grids.level = kwargs.pop('grid_level')
        grids.build()
        scf_obj.grids = grids

    if 'dynamic_level_shift' in kwargs and kwargs.get('dynamic_level_shift'):
        if 'level_shift_factor' in kwargs:
            lev_shift_factor = kwargs.pop('level_shift_factor')
            scf.addons.dynamic_level_shift_(scf_obj, lev_shift_factor)
        else:
            scf.addons.dynamic_level_shift_(scf_obj)

    if 'newton' in kwargs and kwargs.pop('newton'):
        scf_obj = scf.newton(scf_obj)

    if 'fast_newton' in kwargs and kwargs.pop('fast_newton'):
        scf_obj.use_fast_newton = True

    if 'frac_occ' in kwargs and kwargs.get('frac_occ'):
        scf_obj = scf.addons.frac_occ(mf)

    if 'remove_linear_dep' in kwargs and kwargs.get('remove_linear_dep'):
        scf_obj = scf_obj.apply(scf.addons.remove_linear_dep)

    if 'density_fitting' in kwargs and kwargs.get('density_fitting'):
        scf_obj = scf_obj.density_fit()

    if 'excited' in kwargs:
        scf_obj.excited = kwargs['excited']
        #Setup excited object.
        pass

    for key in kwargs:
        setattr(scf_obj, key, kwargs[key])

    return scf_obj

def get_nuc(mol):
    '''Part of the nuclear gradients of core Hamiltonian'''
    if mol._pseudo:
        NotImplementedError('Nuclear gradients for GTH PP')
    else:
        h = mol.intor('int1e_ipnuc', comp=3)
    if mol.has_ecp():
        h += mol.intor('ECPscalar_ipnuc', comp=3)
    return -h


def nuc_grad_generator(mf, mol=None):
    if mol is None: mol = mf.mol
    with_x2c = getattr(mf.base, 'with_x2c', None)
    if with_x2c:
        nuc_deriv = with_x2c.hcore_deriv_generator(deriv=1)
    else:
        with_ecp = mol.has_ecp()
        if with_ecp:
            ecp_atoms = set(mol._ecpbas[:,gto.ATOM_OF])
        else:
            ecp_atoms = ()
        aoslices = mol.aoslice_by_atom()
        h1 = get_nuc(mol)
        def nuc_deriv(atm_id):
            shl0, shl1, p0, p1 = aoslices[atm_id]
            with mol.with_rinv_at_nucleus(atm_id):
                vrinv = mol.intor('int1e_iprinv', comp=3) # <\nabla|1/r|>
                vrinv *= -mol.atom_charge(atm_id)
                if with_ecp and atm_id in ecp_atoms:
                    vrinv += mol.intor('ECPscalar_iprinv', comp=3)
            vrinv[:,p0:p1] += h1[:,p0:p1]
            return vrinv + vrinv.transpose(0,2,1)
    return nuc_deriv

def get_emb_ao(supersystem, subsystem_indices):
    slice_tuple = [0,0,0,0,0,0,0,0]
    for i in range(subsystem_indices[0]):
        slice_tuple[0] += supersystem.subsystems[i].mol.nbas
    slice_tuple[1] = slice_tuple[0] + supersystem.subsystems[subsystem_indices[0]].mol.nbas
    for j in range(subsystem_indices[1]):
        slice_tuple[2] += supersystem.subsystems[j].mol.nbas
    slice_tuple[3] = slice_tuple[2] + supersystem.subsystems[subsystem_indices[1]].mol.nbas
    for k in range(subsystem_indices[2]):
        slice_tuple[4] += supersystem.subsystems[k].mol.nbas
    slice_tuple[5] = slice_tuple[4] + supersystem.subsystems[subsystem_indices[2]].mol.nbas
    for l in range(subsystem_indices[3]):
        slice_tuple[6] += supersystem.subsystems[k].mol.nbas
    slice_tuple[7] = slice_tuple[6] + supersystem.subsystems[subsystem_indices[3]].mol.nbas
    slice_tuple = tuple(slice_tuple)
    return supersystem.mol.intor('int2e', shls_slice=slice_tuple)

def get_emb_mo(supersystem, subsystem_indices):
    #THIS TAKES FOREVER.
    emb_ao = get_emb_ao(supersystem, subsystem_indices)
    mo_coeff_i = supersystem.subsystems[subsystem_indices[0]].env_mo_coeff
    mo_coeff_j = supersystem.subsystems[subsystem_indices[1]].env_mo_coeff
    mo_coeff_k = supersystem.subsystems[subsystem_indices[2]].env_mo_coeff
    mo_coeff_l = supersystem.subsystems[subsystem_indices[3]].env_mo_coeff
    #import time
    #start = time.time()
    emb_mo_a = np.einsum('mnzs,mi,nj,zk,sl', emb_ao, mo_coeff_i[0], mo_coeff_j[0], mo_coeff_k[0], mo_coeff_l[0], optimize=True)
    emb_mo_b = np.einsum('mnzs,mi,nj,zk,sl', emb_ao, mo_coeff_i[1], mo_coeff_j[1], mo_coeff_k[1], mo_coeff_l[1], optimize=True)
    emb_mo_ab = np.einsum('mnzs,mi,nj,zk,sl', emb_ao, mo_coeff_i[0], mo_coeff_j[0], mo_coeff_k[1], mo_coeff_l[1], optimize=True)
    emb_mo_ba = np.einsum('mnzs,mi,nj,zk,sl', emb_ao, mo_coeff_i[1], mo_coeff_j[1], mo_coeff_k[0], mo_coeff_l[0], optimize=True)
    #print ('time elapsed')
    #print (time.time()-start)
    #from pyscf import ao2mo
    #start = time.time()
    #emb_mo_a_test = ao2mo.kernel(emb_ao, (mo_coeff_i[0], mo_coeff_j[0], mo_coeff_k[0], mo_coeff_l[0]))
    #emb_mo_a_test = np.einsum('mnzs,mi,nj,zk,sl', emb_ao, mo_coeff_i[0], mo_coeff_j[0], mo_coeff_k[0], mo_coeff_l[0], optimize=True)
    #print ('time elapsed')
    #print (time.time()-start)
    #print (np.max(np.abs(emb_mo_a - emb_mo_a_test)))
    #emb_mo_b = np.einsum('mnzs,mi,nj,zk,sl->ijkl', emb_ao, mo_coeff_i[1], mo_coeff_j[1], mo_coeff_k[1], mo_coeff_l[1])
    return (emb_mo_a, emb_mo_b, emb_mo_ab, emb_mo_ba)

def gen_vind_emb(supersystem):
    num_subsys = len(supersystem.subsystems)
    resp_terms = [[0] * num_subsys] * num_subsys
    for i, sub in enumerate(supersystem.subsystems):
        for j, alt_sub in enumerate(supersystem.subsystems):
            if i == j:
                if (supersystem.unrestricted or supersystem.mol.spin != 0):
                    nao, nmoa = sub.env_mo_coeff[0].shape
                    nmob = sub.env_mo_coeff[1].shape[1]
                    mocca = sub.env_mo_coeff[0][:,sub.env_mo_occ[0]>0]
                    moccb = sub.env_mo_coeff[1][:,sub.env_mo_occ[1]>0]
                    nocca = mocca.shape[1]
                    noccb = moccb.shape[1]
                    
                    sub_vresp_term = sub.env_scf.gen_response(sub.env_mo_coeff, sub.env_mo_occ, hermi=1)

                    proj_vresp_term = supersystem.gen_proj_response(i,j)

                    def fx(mo1):
                        mo1 = mo1.reshape(-1, nmoa*nocca+nmob*noccb)
                        nset = len(mo1)
                        dm1 = np.empty((2,nset,nao,nao))
                        for i, x in enumerate(mol):
                            xa = x[:nmoa*nocca].reshape(nmoa,nocca)
                            xb = x[nmoa*nocca:].reshape(nmob,noccb)
                            dma = reduce(np.dot, (sub.env_mo_coeff[0], xa, mocca.T))
                            dma = reduce(np.dot, (sub.env_mo_coeff[1], xb, moccb.T))
                            dm1[0,i] = dma + dma.T
                            dm1[1,i] = dmb + dmb.T
                        v1 = sub_vresp_term(dm1) + 2.*proj_vresp_term(dm1)
                        v1vo = np.empty_like(mo1)
                        for i in range(nset):
                            v1vo[i,:nmoa*nocca] = reduce(np.dot, (mo_coeff[0].T, v1[0,i], mocca)).ravel()
                            v1vo[i,nmoa*nocca:] = reduce(np.dot, (mo_coeff[1].T, v1[1,i], moccb)).ravel()
                        return v1vo

                else:
                    nao,nmo = sub.env_mo_coeff[0].shape
                    mocc = sub.env_mo_coeff[0][:,sub.env_mo_occ[0]>0]
                    nocc = mocc.shape[1]
                    sub_vresp_term = sub.env_scf.gen_response(sub.env_mo_coeff[0], sub.env_mo_occ[0]+sub.env_mo_occ[1], hermi=1)

                    proj_vresp_term = supersystem.gen_proj_response(i,j)
                    def fx(mo1):
                        mo1 = mo1.reshape(-1, nmo, nocc)
                        nset = len(mo1)
                        dm1 = np.empty((nset, nao, nao))
                        for i, x in enumerate(mo1):
                            dm = reduce(np.dot, (sub.env_mo_coeff[0], x*2., mocc.T))
                            dm1[i] = dm + dm.T
                        v1 = sub_vresp_term(dm1) + 2.*proj_vresp_term(dm1)
                        v1vo = np.empty_like(mo1)
                        for i, x in enumerate(v1):
                            v1vo[i] = reduce(np.dot, (sub.env_mo_coeff[0].T, x, mocc))
                        return v1vo

            else:
                if (supersystem.unrestricted or supersystem.mol.spin != 0):
                    nao, nmoa = sub.env_mo_coeff[0].shape
                    nmob = sub.env_mo_coeff[1].shape[1]
                    mocca = sub.env_mo_coeff[0][:,sub.env_mo_occ[0]>0]
                    moccb = sub.env_mo_coeff[1][:,sub.env_mo_occ[1]>0]
                    nocca = mocca.shape[1]
                    noccb = moccb.shape[1]
                   
                    fs_emb_mo_coeff = np.zeros_like(supersystem.get_emb_dmat())
                    fs_emb_mo_occ = [np.zeros_like(fs_emb_mo_coeff[0].shape[0]), np.zeros_like(fs_emb_mo_coeff[1].shape[0])] 
                    s2s = supersystem.sub2sup
                    fs_emb_mo_coeff[0][np.ix_(s2s[j],s2s[j])] += alt_sub.env_mo_coeff[0]
                    fs_emb_mo_coeff[1][np.ix_(s2s[j],s2s[j])] += alt_sub.env_mo_coeff[1]
                    fs_emb_mo_occ[0][np.ix_(s2s[j])] += alt_sub.env_mo_occ[0]
                    fs_emb_mo_occ[1][np.ix_(s2s[j])] += alt_sub.env_mo_occ[1]
                    sup_vresp_term = supersystem.fs_scf_obj.gen_response(fs_emb_mo_coeff, fs_emb_mo_occ, hermi=1)

                    proj_vresp_term = supersystem.gen_proj_response(i,j)


                    def fx(mo1):
                        mo1 = mo1.reshape(-1, nmoa*nocca+nmob*noccb)
                        nset = len(mo1)
                        dm1 = np.empty((2,nset,nao,nao))
                        for i, x in enumerate(mol):
                            xa = x[:nmoa*nocca].reshape(nmoa,nocca)
                            xb = x[nmoa*nocca:].reshape(nmob,noccb)
                            dma = reduce(np.dot, (sub.env_mo_coeff[0], xa, mocca.T))
                            dma = reduce(np.dot, (sub.env_mo_coeff[1], xb, moccb.T))
                            dm1[0,i] = dma + dma.T
                            dm1[1,i] = dmb + dmb.T
                        v1 = sup_vresp_term(dm1)[np.ix_([0,1],s2s[i],s2s[i])] + 2.*proj_vresp_term(dm1)
                        v1vo = np.empty_like(mo1)
                        for i in range(nset):
                            v1[0,i] += 2. * reduce(np.dot, (supersystem.fock[0][np.ix_(s2s[i],s2s[j])], dm1[0], supersystem.smat[np.ix_(s2s[j],s2s[i])]))
                            v1[1,i] += 2. * reduce(np.dot, (supersystem.fock[1][np.ix_(s2s[i],s2s[j])], dm1[1], supersystem.smat[np.ix_(s2s[j],s2s[i])]))
                            v1vo[i,:nmoa*nocca] = reduce(np.dot, (mo_coeff[0].T, v1[0,i], mocca)).ravel()
                            v1vo[i,nmoa*nocca:] = reduce(np.dot, (mo_coeff[1].T, v1[1,i], moccb)).ravel()
                        return v1vo

                else:
                    nao,nmo = sub.env_mo_coeff[0].shape
                    mocc = sub.env_mo_coeff[0][:,sub.env_mo_occ[0]>0]
                    nocc = mocc.shape[1]
                    fs_emb_mo_coeff = np.zeros_like(supersystem.get_emb_dmat())
                    fs_emb_mo_occ = np.zeros((supersystem.get_emb_dmat().shape[0]))
                    s2s = supersystem.sub2sup
                    fs_emb_mo_coeff[np.ix_(s2s[j],s2s[j])] += alt_sub.env_mo_coeff[0]
                    fs_emb_mo_occ[np.ix_(s2s[j])] += alt_sub.env_mo_occ[0]
                    fs_emb_mo_occ[np.ix_(s2s[j])] += alt_sub.env_mo_occ[1]
                    sup_vresp_term = supersystem.fs_scf_obj.gen_response(fs_emb_mo_coeff, fs_emb_mo_occ, hermi=1)
                    proj_vresp_term = supersystem.gen_proj_response(i,j)
                    def fx(mo1):
                        mo1 = mo1.reshape(-1, nmo, nocc)
                        nset = len(mo1)
                        dm1 = np.empty((nset, nao, nao))
                        for i, x in enumerate(mo1):
                            dm = reduce(np.dot, (sub.env_mo_coeff[0], x*2., mocc.T))
                            dm1[i] = dm + dm.T
                        v1 = sub_vresp_term(dm1) + 2.*proj_vresp_term(dm1)
                        v1 += 2. * reduce(np.dot, (supersystem.fock[0][np.ix_(s2s[i],s2s[j])], dm1, supersystem.smat[np.ix_(s2s[j],s2s[i])]))
                        v1vo = np.empty_like(mo1)
                        for i, x in enumerate(v1):
                            v1vo[i] = reduce(np.dot, (sub.env_mo_coeff[0].T, x, mocc))
                        return v1vo

            resp_terms[i][j] = fx

    return resp_terms

def dm_cache_xc_kernel(ni, mol, grids, xc_code, dm, spin=0, max_memory=2000):
    '''Compute the 0th order density, Vxc and fxc.  They can be used in TDDFT,
    DFT hessian module etc. This uses the density matrix, not the mo_coeffs.
    '''
    xctype = ni._xc_type(xc_code)
    ao_deriv = 0
    if xctype == 'GGA':
        ao_deriv = 1
    elif xctype == 'NLC':
        raise NotImplementedError('NLC')
    elif xctype == 'MGGA':
        raise NotImplementedError('meta-GGA')

    if spin == 0:
        nao = dm.shape[0]
        rho = []
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory=max_memory):
            rho.append(ni.eval_rho(mol, ao, dm, mask, xctype))
        rho = numpy.hstack(rho)
    else:
        nao = dm[0].shape[0]
        rhoa = []
        rhob = []
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rhoa.append(ni.eval_rho(mol, ao, dm[0], mask, xctype))
            rhob.append(ni.eval_rho(mol, ao, dm[1], mask, xctype))
        rho = (numpy.hstack(rhoa), numpy.hstack(rhob))
    vxc, fxc = ni.eval_xc(xc_code, rho, spin=spin, relativity=0, deriv=2,
                          verbose=0)[1:3]
    return rho, vxc, fxc

