
from pyscf import lib, gto
from pyscf.lib import logger
from pyscf.scf import hf,rohf,uhf
from pyscf.scf import jk
from pyscf.dft import rks, roks, uks, gen_grid, radi
from pyscf.grad import rks as rks_grad

#from pyscf.dft import libxc, numint
from pyscf.dft import numint

import numpy as np
import scipy as sp

from copy import deepcopy as copy

#Custom PYSCF method for the active subsystem.

def get_hcore(mf, emb_pot=None, proj_pot=None, mol=None):
    if mol is None:
        mol = mf.mol
    emb_pot_comb = (emb_pot[0] + emb_pot[1]) / 2.
    proj_pot_comb = (proj_pot[0] + proj_pot[1]) / 2.
    h = mol.intor_symmetric('int1e_kin')
    h+= mol.intor_symmetric('int1e_nuc')
    h+= emb_pot_comb + proj_pot_comb
    return h

#RHF Methods
def rhf_get_fock(mf, emb_pot=None, proj_pot=None, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
             diis_start_cycle=None, level_shift_factor=None, damp_factor=None):
    '''F = h^{core} + V^{HF}
    Special treatment (damping, DIIS, or level shift) will be applied to the
    Fock matrix if diis and cycle is specified (The two parameters are passed
    to get_fock function during the SCF iteration)
    Kwargs:
        h1e : 2D ndarray
            Core hamiltonian
        s1e : 2D ndarray
            Overlap matrix, for DIIS
        vhf : 2D ndarray
            HF potential matrix
        dm : 2D ndarray
            Density matrix, for DIIS
        cycle : int
            Then present SCF iteration step, for DIIS
        diis : an object of :attr:`SCF.DIIS` class
            DIIS object to hold intermediate Fock and error vectors
        diis_start_cycle : int
            The step to start DIIS.  Default is 0.
        level_shift_factor : float or int
            Level shift (in AU) for virtual space.  Default is 0.
    '''
    if emb_pot is None: emb_pot = 0.0
    if proj_pot is None: proj_pot = 0.0
    if h1e is None: h1e = mf.get_hcore()
    if vhf is None: vhf = mf.get_veff(dm=dm)
    f = h1e + vhf + emb_pot + proj_pot #Added embedding potential to fock
    #return f

    if cycle < 0 and diis is None:  # Not inside the SCF iteration
        return f

    if diis_start_cycle is None:
        diis_start_cycle = mf.diis_start_cycle
    if level_shift_factor is None:
        level_shift_factor = mf.level_shift
    if damp_factor is None:
        damp_factor = mf.damp
    if s1e is None: s1e = mf.get_ovlp()
    if dm is None: dm = self.make_rdm1()

    if 0 <= cycle < diis_start_cycle-1 and abs(damp_factor) > 1e-4:
        f = damping(s1e, dm*.5, f, damp_factor)
    if diis is not None and cycle >= diis_start_cycle:
        f = diis.update(s1e, dm, f, mf, h1e, vhf)
    if abs(level_shift_factor) > 1e-4:
        f = level_shift(s1e, dm*.5, f, level_shift_factor)
    return f

def rhf_energy_elec(mf, emb_pot=None, proj_pot=None, dm=None, h1e=None, vhf=None):

    if emb_pot is None: emb_pot = 0.0
    if dm is None: dm = mf.make_rdm1()
    if h1e is None: h1e = mf.get_hcore()
    if vhf is None: vhf = mf.get_veff(mf.mol, dm)
    h1e = copy(h1e + proj_pot + emb_pot) #Add embedding potential to the core ham
    vhf = copy(vhf) #+ emb_pot) #Add embedding potential to the core ham
    e1 = np.einsum('ij,ji', h1e, dm).real
    e_coul = np.einsum('ij,ji', vhf, dm).real * .5
    logger.debug(mf, 'E_coul = %.15g', e_coul)
    return e1+e_coul, e_coul

#ROHF Methods
def rohf_get_fock(mf, emb_pot=None, proj_pot=None, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
             diis_start_cycle=None, level_shift_factor=None, damp_factor=None):
    '''Build fock matrix based on Roothaan's effective fock.
    See also :func:`get_roothaan_fock`
    '''
    if emb_pot is None: emb_pot = [0.0, 0.0]
    if h1e is None: h1e = mf.get_hcore()
    if s1e is None: s1e = mf.get_ovlp()
    if vhf is None: vhf = mf.get_veff(dm=dm)
    if dm is None: dm = mf.make_rdm1()
    if isinstance(dm, np.ndarray) and dm.ndim == 2:
        dm = np.array((dm*.5, dm*.5))
# To Get orbital energy in get_occ, we saved alpha and beta fock, because
# Roothaan effective Fock cannot provide correct orbital energy with `eig`
# TODO, check other treatment  J. Chem. Phys. 133, 141102
    focka = h1e + vhf[0] + emb_pot[0] + proj_pot[0]#Add embedding potential
    fockb = h1e + vhf[1] + emb_pot[1] + proj_pot[1]#Add embedding potential
    f = rohf.get_roothaan_fock((focka,fockb), dm, s1e)
    if cycle < 0 and diis is None:  # Not inside the SCF iteration
        return f

    if diis_start_cycle is None:
        diis_start_cycle = mf.diis_start_cycle
    if level_shift_factor is None:
        level_shift_factor = mf.level_shift
    if damp_factor is None:
        damp_factor = mf.damp

    dm_tot = dm[0] + dm[1]
    if 0 <= cycle < diis_start_cycle-1 and abs(damp_factor) > 1e-4:
        raise NotImplementedError('ROHF Fock-damping')
    if diis and cycle >= diis_start_cycle:
        f = diis.update(s1e, dm_tot, f, mf, h1e, vhf)
    if abs(level_shift_factor) > 1e-4:
        f = hf.level_shift(s1e, dm_tot*.5, f, level_shift_factor)
    f = lib.tag_array(f, focka=focka, fockb=fockb)
    return f

def rohf_energy_elec(mf, emb_pot=None, proj_pot=None, dm=None, h1e=None, vhf=None):

    if emb_pot is None: emb_pot = [0.0, 0.0]
    if dm is None: dm = mf.make_rdm1()
    elif isinstance(dm, np.ndarray) and dm.ndim == 2:
        dm = np.array((dm*.5, dm*.5))
    ee, ecoul = uhf_energy_elec(mf, emb_pot, proj_pot, dm, h1e, vhf)
    logger.debug(mf, 'Ecoul = %.15g', ecoul)
    return ee, ecoul


#UHF Methods
def uhf_get_fock(mf, emb_pot=None, proj_pot=None, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
             diis_start_cycle=None, level_shift_factor=None, damp_factor=None):


    if emb_pot is None: emb_pot = [0.0, 0.0]
    if proj_pot is None: proj_pot = [0.0, 0.0]
    if h1e is None: h1e = mf.get_hcore()
    #if vhf is None: vhf = mf.get_veff(dm=dm)
    #For some reason the vhf being passed is wrong I believe.
    vhf = mf.get_veff(dm=dm)
    f = h1e + vhf 
    if f.ndim == 2:
        f = (f, f)
    f[0] = f[0] + emb_pot[0] + proj_pot[0] #Add embedding potential
    f[1] = f[1] + emb_pot[1] + proj_pot[1] #Add embedding potential


    if cycle < 0 and diis is None:  # Not inside the SCF iteration
        return f

    if diis_start_cycle is None:
        diis_start_cycle = mf.diis_start_cycle
    if level_shift_factor is None:
        level_shift_factor = mf.level_shift
    if damp_factor is None:
        damp_factor = mf.damp
    if s1e is None: s1e = mf.get_ovlp()
    if dm is None: dm = self.make_rdm1()

    if isinstance(level_shift_factor, (tuple, list, np.ndarray)):
        shifta, shiftb = level_shift_factor
    else:
        shifta = shiftb = level_shift_factor
    if isinstance(damp_factor, (tuple, list, np.ndarray)):
        dampa, dampb = damp_factor
    else:
        dampa = dampb = damp_factor

    if isinstance(dm, np.ndarray) and dm.ndim == 2:
        dm = [dm*.5] * 2
    if 0 <= cycle < diis_start_cycle-1 and abs(dampa)+abs(dampb) > 1e-4:
        f = (hf.damping(s1e, dm[0], f[0], dampa),
             hf.damping(s1e, dm[1], f[1], dampb))
    if diis and cycle >= diis_start_cycle:
        f = diis.update(s1e, dm, f, mf, h1e, vhf)
    if abs(shifta)+abs(shiftb) > 1e-4:
        f = (hf.level_shift(s1e, dm[0], f[0], shifta),
             hf.level_shift(s1e, dm[1], f[1], shiftb))
    return np.array(f)

def uhf_energy_elec(mf, emb_pot=None, proj_pot=None, dm=None, h1e=None, vhf=None):
    '''Electronic energy of Unrestricted Hartree-Fock
    Returns:
        Hartree-Fock electronic energy and the 2-electron part contribution
    '''
    if emb_pot is None: emb_pot = [0.0, 0.0]
    if proj_pot is None: proj_pot = [0.0, 0.0]
    if dm is None: dm = mf.make_rdm1()
    if h1e is None:
        h1e = mf.get_hcore()
    if isinstance(dm, np.ndarray) and dm.ndim == 2:
        dm = np.array((dm*.5, dm*.5))
    if vhf is None:
        vhf = mf.get_veff(mf.mol, dm)
    e1 = np.einsum('ij,ij', h1e.conj(), dm[0]+dm[1])

    new_vhf = [None, None]
    new_vhf[0] = vhf[0] + 2.*(emb_pot[0] + proj_pot[0])
    new_vhf[1] = vhf[1] + 2.*(emb_pot[1] + proj_pot[1])

    e_coul =(np.einsum('ij,ji', new_vhf[0], dm[0]) +
             np.einsum('ij,ji', new_vhf[1], dm[1])).real * .5
    return e1+e_coul, e_coul

#RKS Methods
def rks_energy_elec(ks, emb_pot=None, proj_pot=None, dm=None, h1e=None, vhf=None):
    r'''Electronic part of RKS energy.
    Args:
        ks : an instance of DFT class
        dm : 2D ndarray
            one-partical density matrix
        h1e : 2D ndarray
            Core hamiltonian
    Returns:
        RKS electronic energy and the 2-electron part contribution
    '''
    if emb_pot is None: emb_pot = 0.0
    if proj_pot is None: proj_pot = 0.0
    if dm is None: dm = ks.make_rdm1()
    if h1e is None: h1e = ks.get_hcore()
    if vhf is None or getattr(vhf, 'ecoul', None) is None:
        vhf = ks.get_veff(ks.mol, dm)

    h1e = h1e + emb_pot + proj_pot
    e1 = np.einsum('ij,ji', h1e, dm).real
    tot_e = e1 + vhf.ecoul + vhf.exc
    logger.debug(ks, 'Ecoul = %s  Exc = %s', vhf.ecoul, vhf.exc)
    return tot_e, vhf.ecoul+vhf.exc

rks_get_fock = rhf_get_fock


#UKS Methods
def uks_energy_elec(ks, emb_pot=None, proj_pot=None, dm=None, h1e=None, vhf=None):
    if emb_pot is None: emb_pot = [0.0,0.0]
    if dm is None: dm = ks.make_rdm1()
    if h1e is None: h1e = ks.get_hcore()
    if vhf is None or getattr(vhf, 'ecoul', None) is None:
        vhf = ks.get_veff(ks.mol, dm)
    if isinstance(dm, np.ndarray) and dm.ndim == 2:
        dm = np.array((dm*.5, dm*.5))
    emb_h1e = [None,None] 
    emb_h1e[0] = h1e + emb_pot[0] + proj_pot[0]
    emb_h1e[1] = h1e + emb_pot[1] + proj_pot[1]
    e1 = np.einsum('ij,ji', emb_h1e[0], dm[0]) + np.einsum('ij,ji', emb_h1e[1], dm[1])
    tot_e = e1.real + vhf.ecoul + vhf.exc
    logger.debug(ks, 'Ecoul = %s  Exc = %s', vhf.ecoul, vhf.exc)
    return tot_e, vhf.ecoul+vhf.exc

uks_get_fock = uhf_get_fock

#ROKS Methods
roks_energy_elec = uks_energy_elec
roks_get_fock = rohf_get_fock

def exc_rks(mf, emb_dm, dm, relativity=0, hermi=0,
           max_memory=2000, verbose=None):

    ni = mf._numint
    mol = mf.mol
    grids = mf.grids
    xc_code = mf.xc
    

    xctype = ni._xc_type(xc_code)
    make_rho_emb, nset, nao = ni._gen_rho_evaluator(mol, emb_dm, hermi)
    make_rho_sub, nset_2, nao_2 = ni._gen_rho_evaluator(mol, dm, hermi)

    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    nelec = np.zeros(nset)
    excsum = np.zeros(nset)
    vmat = np.zeros((nset,nao,nao))
    aow = None

    if xctype == 'LDA':
        ao_deriv = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            aow = np.ndarray(ao.shape, order='F', buffer=aow)
            for idm in range(nset):
                rho_emb = make_rho_emb(idm, ao, mask, 'LDA')
                rho_sub = make_rho_sub(idm, ao, mask, 'LDA')
                exc, vxc = ni.eval_xc(xc_code, rho_emb, 0, relativity, 1, verbose)[:2]
                vrho = vxc[0]
                den = rho_sub * weight
                nelec[idm] += den.sum()
                excsum[idm] += np.dot(den, exc)
                # *.5 because vmat + vmat.T
                aow = np.einsum('pi,p->pi', ao, .5*weight*vrho, out=aow)
                vmat[idm] += numint._dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
                rho_emb = rho_sub = exc = vxc = vrho = None
    elif xctype == 'GGA':
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            ngrid = weight.size
            aow = np.ndarray(ao[0].shape, order='F', buffer=aow)
            for idm in range(nset):
                rho_emb = make_rho_emb(idm, ao, mask, 'GGA')
                rho_sub = make_rho_sub(idm, ao, mask, 'GGA')
                exc, vxc = ni.eval_xc(xc_code, rho_emb, 0, relativity, 1, verbose)[:2]
                den = rho_sub[0] * weight
                nelec[idm] += den.sum()
                excsum[idm] += np.dot(den, exc)
# ref eval_mat function
                wv = numint._rks_gga_wv0(rho_sub, vxc, weight)
                aow = np.einsum('npi,np->pi', ao, wv, out=aow)
                vmat[idm] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
                rho_emb = rho_sub = exc = vxc = wv = None
    elif xctype == 'NLC':
        nlc_pars = ni.nlc_coeff(xc_code[:-6])
        if nlc_pars == [0,0]:
            raise NotImplementedError('VV10 cannot be used with %s. '
                                      'The supported functionals are %s' %
                                      (xc_code[:-6], ni.libxc.VV10_XC))
        ao_deriv = 1
        vvrho=np.empty([nset,4,0])
        vvweight=np.empty([nset,0])
        vvcoords=np.empty([nset,0,3])
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rhotmp = np.empty([0,4,weight.size])
            weighttmp = np.empty([0,weight.size])
            coordstmp = np.empty([0,weight.size,3])
            for idm in range(nset):
                rho_emb = make_rho_emb(idm, ao, mask, 'GGA')
                rho_sub = make_rho_sub(idm, ao, mask, 'GGA')
                rho_emb = np.expand_dims(rho,axis=0)
                rhotmp = np.concatenate((rhotmp,rho_emb),axis=0)
                weighttmp = np.concatenate((weighttmp,np.expand_dims(weight,axis=0)),axis=0)
                coordstmp = np.concatenate((coordstmp,np.expand_dims(coords,axis=0)),axis=0)
                rho = None
            vvrho=np.concatenate((vvrho,rhotmp),axis=2)
            vvweight=np.concatenate((vvweight,weighttmp),axis=1)
            vvcoords=np.concatenate((vvcoords,coordstmp),axis=1)
            rhotmp = weighttmp = coordstmp = None
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            ngrid = weight.size
            aow = np.ndarray(ao[0].shape, order='F', buffer=aow)
            for idm in range(nset):
                rho_emb = make_rho_emb(idm, ao, mask, 'GGA')
                rho_sub = make_rho_sub(idm, ao, mask, 'GGA')
                exc, vxc = numint._vv10nlc(rho_emb,coords,vvrho[idm],vvweight[idm],vvcoords[idm],nlc_pars)
                den = rho_sub[0] * weight
                nelec[idm] += den.sum()
                excsum[idm] += np.dot(den, exc)
# ref eval_mat function
                wv = numint._rks_gga_wv0(rho_emb, vxc, weight)
                aow = np.einsum('npi,np->pi', ao, wv, out=aow)
                vmat[idm] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
                rho = exc = vxc = wv = None
        vvrho = vvweight = vvcoords = None
    elif xctype == 'MGGA':
        if (any(x in xc_code.upper() for x in ('CC06', 'CS', 'BR89', 'MK00'))):
            raise NotImplementedError('laplacian in meta-GGA method')
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            ngrid = weight.size
            aow = np.ndarray(ao[0].shape, order='F', buffer=aow)
            for idm in range(nset):
                rho_emb = make_rho_emb(idm, ao, mask, 'MGGA')
                rho_sub = make_rho_sub(idm, ao, mask, 'MGGA')
                exc, vxc = ni.eval_xc(xc_code, rho_emb, 0, relativity, 1, verbose)[:2]
                vrho, vsigma, vlapl, vtau = vxc[:4]
                den = rho_sub[0] * weight
                nelec[idm] += den.sum()
                excsum[idm] += np.dot(den, exc)

                wv = numint._rks_gga_wv0(rho_emb, vxc, weight)
                aow = np.einsum('npi,np->pi', ao[:4], wv, out=aow)
                vmat[idm] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)

# FIXME: .5 * .5   First 0.5 for v+v.T symmetrization.
# Second 0.5 is due to the Libxc convention tau = 1/2 \nabla\phi\dot\nabla\phi
                wv = (.5 * .5 * weight * vtau).reshape(-1,1)
                vmat[idm] += numint._dot_ao_ao(mol, ao[1], wv*ao[1], mask, shls_slice, ao_loc)
                vmat[idm] += numint._dot_ao_ao(mol, ao[2], wv*ao[2], mask, shls_slice, ao_loc)
                vmat[idm] += numint._dot_ao_ao(mol, ao[3], wv*ao[3], mask, shls_slice, ao_loc)

                rho_emb = rho_sub = exc = vxc = vrho = vsigma = wv = None

    for i in range(nset):
        vmat[i] = vmat[i] + vmat[i].T
    if nset == 1:
        nelec = nelec[0]
        excsum = excsum[0]
        vmat = vmat.reshape(nao,nao)
    return nelec, excsum, vmat

def exc_uks(mf, emb_dm, dms, relativity=0, hermi=0,
           max_memory=2000, verbose=None):

    ni = mf._numint
    mol = mf.mol
    grid = mf.grids
    xc_code = mf.xc


    xctype = ni._xc_type(xc_code)
    if xctype == 'NLC':
        dms_sf = dms[0] + dms[1]
        nelec, excsum, vmat = exc_rks(mf, emb_dm, dms, relativity, hermi,
                                     max_memory, verbose)
        return [nelec,nelec], excsum, np.asarray([vmat,vmat])

    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    dma, dmb = _format_uks_dm(dms)
    nao = dma.shape[-1]
    make_rhoa_emb, nset = ni._gen_rho_evaluator(mol, emb_dm[0], hermi)[:2]
    make_rhob_emb      = ni._gen_rho_evaluator(mol, emb_dm[1], hermi)[0]
    make_rhoa_sub      = ni._gen_rho_evaluator(mol, dms[0], hermi)[0]
    make_rhob_sub      = ni._gen_rho_evaluator(mol, dms[1], hermi)[0]

    nelec = np.zeros((2,nset))
    excsum = np.zeros(nset)
    vmat = np.zeros((2,nset,nao,nao))
    aow = None
    if xctype == 'LDA':
        ao_deriv = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            aow = np.ndarray(ao.shape, order='F', buffer=aow)
            for idm in range(nset):
                rho_a_emb = make_rhoa_emb(idm, ao, mask, xctype)
                rho_b_emb = make_rhob_emb(idm, ao, mask, xctype)
                rho_a_sub = make_rhoa_sub(idm, ao, mask, xctype)
                rho_b_sub = make_rhob_sub(idm, ao, mask, xctype)
                exc, vxc = ni.eval_xc(xc_code, (rho_a_emb, rho_b_emb),
                                      1, relativity, 1, verbose)[:2]
                vrho = vxc[0]
                den = rho_a_sub * weight
                nelec[0,idm] += den.sum()
                excsum[idm] += np.dot(den, exc)
                den = rho_b_sub * weight
                nelec[1,idm] += den.sum()
                excsum[idm] += np.dot(den, exc)

                # *.5 due to +c.c. in the end
                aow = np.einsum('pi,p->pi', ao, .5*weight*vrho[:,0], out=aow)
                vmat[0,idm] += numint._dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
                aow = np.einsum('pi,p->pi', ao, .5*weight*vrho[:,1], out=aow)
                vmat[1,idm] += numint._dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
                rho_a_emb = rho_b_emb = rho_a_sub = rho_b_sub = exc = vxc = vrho = None
    elif xctype == 'GGA':
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            ngrid = weight.size
            aow = np.ndarray(ao[0].shape, order='F', buffer=aow)
            for idm in range(nset):
                rho_a_emb = make_rhoa_emb(idm, ao, mask, xctype)
                rho_b_emb = make_rhob_emb(idm, ao, mask, xctype)
                rho_a_sub = make_rhoa_sub(idm, ao, mask, xctype)
                rho_b_sub = make_rhob_sub(idm, ao, mask, xctype)
                exc, vxc = ni.eval_xc(xc_code, (rho_a_emb, rho_b_emb),
                                      1, relativity, 1, verbose)[:2]
                den = rho_a_sub[0]*weight
                nelec[0,idm] += den.sum()
                excsum[idm] += np.dot(den, exc)
                den = rho_b_sub[0]*weight
                nelec[1,idm] += den.sum()
                excsum[idm] += np.dot(den, exc)

                wva, wvb = _uks_gga_wv0((rho_a_emb,rho_b_emb), vxc, weight)
                aow = np.einsum('npi,np->pi', ao, wva, out=aow)
                vmat[0,idm] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
                aow = np.einsum('npi,np->pi', ao, wvb, out=aow)
                vmat[1,idm] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
                rho_a_emb = rho_b_emb = rho_a_sub = rho_b_sub = exc = vxc = wva = wvb = None
    elif xctype == 'MGGA':
        if (any(x in xc_code.upper() for x in ('CC06', 'CS', 'BR89', 'MK00'))):
            raise NotImplementedError('laplacian in meta-GGA method')
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            ngrid = weight.size
            aow = np.ndarray(ao[0].shape, order='F', buffer=aow)
            for idm in range(nset):
                rho_a_emb = make_rhoa_emb(idm, ao, mask, xctype)
                rho_b_emb = make_rhob_emb(idm, ao, mask, xctype)
                rho_a_sub = make_rhoa_sub(idm, ao, mask, xctype)
                rho_b_sub = make_rhob_sub(idm, ao, mask, xctype)
                exc, vxc = ni.eval_xc(xc_code, (rho_a_emb, rho_b_emb),
                                      1, relativity, 1, verbose)[:2]
                vrho, vsigma, vlapl, vtau = vxc[:4]
                den = rho_a_sub[0]*weight
                nelec[0,idm] += den.sum()
                excsum[idm] += np.dot(den, exc)
                den = rho_b_sub[0]*weight
                nelec[1,idm] += den.sum()
                excsum[idm] += np.dot(den, exc)

                wva, wvb = _uks_gga_wv0((rho_a_emb,rho_b_emb), vxc, weight)
                aow = np.einsum('npi,np->pi', ao[:4], wva, out=aow)
                vmat[0,idm] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
                aow = np.einsum('npi,np->pi', ao[:4], wvb, out=aow)
                vmat[1,idm] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)

# FIXME: .5 * .5   First 0.5 for v+v.T symmetrization.
# Second 0.5 is due to the Libxc convention tau = 1/2 \nabla\phi\dot\nabla\phi
                wv = (.25 * weight * vtau[:,0]).reshape(-1,1)
                vmat[0,idm] += numint._dot_ao_ao(mol, ao[1], wv*ao[1], mask, shls_slice, ao_loc)
                vmat[0,idm] += numint._dot_ao_ao(mol, ao[2], wv*ao[2], mask, shls_slice, ao_loc)
                vmat[0,idm] += numint._dot_ao_ao(mol, ao[3], wv*ao[3], mask, shls_slice, ao_loc)
                wv = (.25 * weight * vtau[:,1]).reshape(-1,1)
                vmat[1,idm] += numint._dot_ao_ao(mol, ao[1], wv*ao[1], mask, shls_slice, ao_loc)
                vmat[1,idm] += numint._dot_ao_ao(mol, ao[2], wv*ao[2], mask, shls_slice, ao_loc)
                vmat[1,idm] += numint._dot_ao_ao(mol, ao[3], wv*ao[3], mask, shls_slice, ao_loc)
                rho_a_emb = rho_b_emb = rho_a_sub = rho_b_sub = exc = vxc = vrho = vsigma = wva = wvb = None

    for i in range(nset):
        vmat[0,i] = vmat[0,i] + vmat[0,i].T
        vmat[1,i] = vmat[1,i] + vmat[1,i].T
    if isinstance(dma, np.ndarray) and dma.ndim == 2:
        vmat = vmat[:,0]
        nelec = nelec.reshape(2)
        excsum = excsum[0]
    return nelec, excsum, vmat

def get_veff_grad(ks_grad, mol=None, dm=None):
    '''
    First order derivative of DFT effective potential matrix (wrt electron coordinates).
    '''
    if mol is None: mol = ks_grad.mol
    if dm is None: dm = ks_grad.base.make_rdm1()

    #t0 = (logger.process_clock(), logger.perf_counter())

    mf = ks_grad.base
    ni = mf._numint
    if ks_grad.grids is not None:
        grids = ks_grad.grids
    else:
        grids = mf.grids
    if grids.coords is None:
        grids.build(with_non0tab=True)

    if mf.nlc != '':
        raise NotImplementedError
    #enabling range-separated hybrids
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, ks_grad.max_memory*.9-mem_now)
    if isinstance(mf, rks.RKS):
        exc, vxc = get_rks_subsystem_vxc_full_response(ni, mol, grids, mf.xc, dm,
                                         max_memory=max_memory,
                                         verbose=ks_grad.verbose)
        logger.debug1(ks_grad, 'sum(grids response) %s', exc.sum(axis=0))
        #t0 = logger.timer(ks_grad, 'vxc', *t0)

        if abs(hyb) < 1e-10 and abs(alpha) < 1e-10:
            vj = ks_grad.get_j(mol, dm)
            vxc += vj
        else:
            vj, vk = ks_grad.get_jk(mol, dm)
            vk *= hyb
            if abs(omega) > 1e-10:  # For range separated Coulomb operator
                with mol.with_range_coulomb(omega):
                    vk += ks_grad.get_k(mol, dm) * (alpha - hyb)
            vxc += vj - vk * .5
    else:
        exc, vxc = get_uks_subsystem_vxc_full_response(ni, mol, grids, mf.xc, dm,
                                         max_memory=max_memory,
                                         verbose=ks_grad.verbose)
        logger.debug1(ks_grad, 'sum(grids response) %s', exc.sum(axis=0))
        #t0 = logger.timer(ks_grad, 'vxc', *t0)

        if abs(hyb) < 1e-10:
            vj = ks_grad.get_j(mol, dm)
            vxc += vj[0] + vj[1]
        else:
            vj, vk = ks_grad.get_jk(mol, dm)
            vk *= hyb
            if abs(omega) > 1e-10:  # For range separated Coulomb operator
                with mol.with_range_coulomb(omega):
                    vk += ks_grad.get_k(mol, dm) * (alpha - hyb)
            vxc += vj[0] + vj[1] - vk



    return lib.tag_array(vxc, exc1_grid=exc)

def get_rks_subsystem_vxc_full_response(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
                                        max_memory=2000, verbose=None):
    '''Full response including the response of the grids. The grid response is
    different for a subsystem because of the unique grid for subsystems.'''
    xctype = ni._xc_type(xc_code)
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi)
    ao_loc = mol.ao_loc_nr()

    excsum = 0
    vmat = np.zeros((3,nao,nao))

    if xctype == 'LDA':
        ao_deriv = 1
        vtmp = np.empty((3,nao,nao))
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            for idm in range(nset):
                rho = make_rho(idm, ao[0], mask, 'LDA')
                vxc = ni.eval_xc(xc_code, rho, 0, relativity, 1,
                                 verbose=verbose)[1]
                vrho = vxc[0]
                aow = np.einsum('pi,p->pi', ao[0], weight*vrho)
                rks_grad._d1_dot_(vmat[idm], mol, ao[1:4], aow, mask, ao_loc, True)
                rho = vxc = vrho = aow = None

        for atm_id, (coords, weight, weight1) in enumerate(grids_response_cc(grids)):
            print ('grad atm_id')
            print (atm_id)
            mask = gen_grid.make_mask(mol, coords)
            ao = ni.eval_ao(mol, coords, deriv=ao_deriv, non0tab=mask)
            rho = make_rho(0, ao[0], mask, 'LDA')
            exc, vxc = ni.eval_xc(xc_code, rho, 0, relativity, 1,
                                  verbose=verbose)[:2]
            vrho = vxc[0]

            vtmp = np.zeros((3,nao,nao))
            aow = np.einsum('pi,p->pi', ao[0], weight*vrho)
            rks_grad._d1_dot_(vtmp, mol, ao[1:4], aow, mask, ao_loc, True)

            # response of weights
            #This part can be done for full system.
            excsum += np.einsum('r,r,nxr->nx', exc, rho, weight1)
            # response of grids coordinates
            #This part is only done for atoms in subsystem.
            excsum[atm_id] += np.einsum('xij,ji->x', vtmp, dms) * 2
            print (excsum)
            rho = vxc = vrho = aow = None

    elif xctype == 'GGA':
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            for idm in range(nset):
                rho = make_rho(idm, ao[:4], mask, 'GGA')
                vxc = ni.eval_xc(xc_code, rho, 0, relativity, 1,
                                 verbose=verbose)[1]
                wv = numint._rks_gga_wv0(rho, vxc, weight)
                rks_grad._gga_grad_sum_(vmat[idm], mol, ao, wv, mask, ao_loc)
                rho = vxc = vrho = wv = None

        for atm_id, (coords, weight, weight1) in enumerate(grids_response_cc(grids)):
            print ('grad atm_id')
            print (atm_id)
            mask = gen_grid.make_mask(mol, coords)
            ao = ni.eval_ao(mol, coords, deriv=ao_deriv, non0tab=mask)
            rho = make_rho(0, ao[:4], mask, 'GGA')
            exc, vxc = ni.eval_xc(xc_code, rho, 0, relativity, 1,
                                  verbose=verbose)[:2]

            vtmp = np.zeros((3,nao,nao))
            wv = numint._rks_gga_wv0(rho, vxc, weight)
            rks_grad._gga_grad_sum_(vtmp, mol, ao, wv, mask, ao_loc)

            # response of weights
            excsum += np.einsum('r,r,nxr->nx', exc, rho[0], weight1)
            # response of grids coordinates
            excsum[atm_id] += np.einsum('xij,ji->x', vtmp, dms) * 2
            print (excsum)
            rho = vxc = vrho = wv = None

    elif xctype == 'NLC':
        raise NotImplementedError('NLC')
    elif xctype == 'MGGA':
        raise NotImplementedError('meta-GGA')

    # - sign because nabla_X = -nabla_x
    return excsum, -vmat

def get_uks_subsystem_vxc_full_response(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
                                        max_memory=2000, verbose=None):
    '''Full response including the response of the grids. The grid response is
    different for a subsystem because of the unique grid for subsystems.'''
    '''Full response including the response of the grids'''
    xctype = ni._xc_type(xc_code)
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi)
    ao_loc = mol.ao_loc_nr()
    aoslices = mol.aoslice_by_atom()

    excsum = 0
    vmat = np.zeros((2,3,nao,nao))
    if xctype == 'LDA':
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho_a = make_rho(0, ao[0], mask, 'LDA')
            rho_b = make_rho(1, ao[0], mask, 'LDA')
            vxc = ni.eval_xc(xc_code, (rho_a,rho_b), 1, relativity, 1,
                             verbose=verbose)[1]
            vrho = vxc[0]
            aow = np.einsum('pi,p->pi', ao[0], weight*vrho[:,0])
            rks_grad._d1_dot_(vmat[0], mol, ao[1:4], aow, mask, ao_loc, True)
            aow = np.einsum('pi,p->pi', ao[0], weight*vrho[:,1])
            rks_grad._d1_dot_(vmat[1], mol, ao[1:4], aow, mask, ao_loc, True)
            vxc = vrho = aow = None

        for atm_id, (coords, weight, weight1) \
                in enumerate(rks_grad.grids_response_cc(grids)):
            print ('grad atm_id')
            print (atm_id)
            mask = gen_grid.make_mask(mol, coords)
            ao = ni.eval_ao(mol, coords, deriv=ao_deriv, non0tab=mask)
            rho_a = make_rho(0, ao[0], mask, 'LDA')
            rho_b = make_rho(1, ao[0], mask, 'LDA')
            exc, vxc = ni.eval_xc(xc_code, (rho_a,rho_b), 1, relativity, 1,
                                  verbose=verbose)[:2]
            vrho = vxc[0]

            vtmp = np.zeros((3,nao,nao))
            aow = np.einsum('pi,p->pi', ao[0], weight*vrho[:,0])
            rks_grad._d1_dot_(vtmp, mol, ao[1:4], aow, mask, ao_loc, True)
            excsum += np.einsum('r,r,nxr->nx', exc, rho_a+rho_b, weight1)
            excsum[atm_id] += np.einsum('xij,ji->x', vtmp, dms[0]) * 2

            vtmp = np.zeros((3,nao,nao))
            aow = np.einsum('pi,p->pi', ao[0], weight*vrho[:,1])
            rks_grad._d1_dot_(vtmp, mol, ao[1:4], aow, mask, ao_loc, True)
            excsum[atm_id] += np.einsum('xij,ji->x', vtmp, dms[1]) * 2
            print (excsum)
            vxc = vrho = aow = None

    elif xctype == 'GGA':
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            print (weight.shape)
            rho_a = make_rho(0, ao[:4], mask, 'GGA')
            rho_b = make_rho(1, ao[:4], mask, 'GGA')
            vxc = ni.eval_xc(xc_code, (rho_a,rho_b), 1, relativity, 1,
                             verbose=verbose)[1]
            wva, wvb = numint._uks_gga_wv0((rho_a,rho_b), vxc, weight)

            rks_grad._gga_grad_sum_(vmat[0], mol, ao, wva, mask, ao_loc)
            rks_grad._gga_grad_sum_(vmat[1], mol, ao, wvb, mask, ao_loc)
            rho_a = rho_b = vxc = wva = wvb = None

        for atm_id, (coords, weight, weight1) \
                in enumerate(rks_grad.grids_response_cc(grids)):
            print ('grad atm_id')
            print (atm_id)
            print (weight1.shape)
            #I should be able to get the numerical gradient of the weights actually.
            mask = gen_grid.make_mask(mol, coords)
            ao = ni.eval_ao(mol, coords, deriv=ao_deriv, non0tab=mask)
            rho_a = make_rho(0, ao[:4], mask, 'GGA')
            rho_b = make_rho(1, ao[:4], mask, 'GGA')
            exc, vxc = ni.eval_xc(xc_code, (rho_a,rho_b), 1, relativity, 1,
                                  verbose=verbose)[:2]
            wva, wvb = numint._uks_gga_wv0((rho_a,rho_b), vxc, weight)

            vtmp = np.zeros((3,nao,nao))
            rks_grad._gga_grad_sum_(vtmp, mol, ao, wva, mask, ao_loc)
            excsum += np.einsum('r,r,nxr->nx', exc, rho_a[0]+rho_b[0], weight1)
            excsum[atm_id] += np.einsum('xij,ji->x', vtmp, dms[0]) * 2

            vtmp = np.zeros((3,nao,nao))
            rks_grad._gga_grad_sum_(vtmp, mol, ao, wvb, mask, ao_loc)
            excsum[atm_id] += np.einsum('xij,ji->x', vtmp, dms[1]) * 2
            print (excsum)
            rho_a = rho_b = vxc = wva = wvb = None

    elif xctype == 'NLC':
        raise NotImplementedError('NLC')
    elif xctype == 'MGGA':
        raise NotImplementedError('meta-GGA')

    # - sign because nabla_X = -nabla_x
    return excsum, -vmat

def grids_response(grids):
    # JCP 98, 5612 (1993); DOI:10.1063/1.464906
    mol = grids.mol
    atom_grids_tab = grids.gen_atomic_grids(mol, grids.atom_grid,
                                            grids.radi_method,
                                            grids.level, grids.prune)
    atm_coords = np.asarray(mol.atom_coords() , order='C')
    atm_dist = gto.mole.inter_distance(mol, atm_coords)

    def _radii_adjust(mol, atomic_radii):
        charges = mol.atom_charges()
        if grids.radii_adjust == radi.treutler_atomic_radii_adjust:
            rad = np.sqrt(atomic_radii[charges]) + 1e-200
        elif grids.radii_adjust == radi.becke_atomic_radii_adjust:
            rad = atomic_radii[charges] + 1e-200
        else:
            fadjust = lambda i, j, g: g
            gadjust = lambda *args: 1
            return fadjust, gadjust

        rr = rad.reshape(-1,1) * (1./rad)
        a = .25 * (rr.T - rr)
        a[a<-.5] = -.5
        a[a>0.5] = 0.5

        def fadjust(i, j, g):
            return g + a[i,j]*(1-g**2)

        #: d[g + a[i,j]*(1-g**2)] /dg = 1 - 2*a[i,j]*g
        def gadjust(i, j, g):
            return 1 - 2*a[i,j]*g
        return fadjust, gadjust

    fadjust, gadjust = _radii_adjust(mol, grids.atomic_radii)

    def gen_grid_partition(coords, atom_id):
        ngrids = coords.shape[0]
        grid_dist = np.empty((mol.natm,ngrids))
        for ia in range(mol.natm):
            dc = coords - atm_coords[ia]
            grid_dist[ia] = np.linalg.norm(dc,axis=1) + 1e-200

        pbecke = np.ones((mol.natm,ngrids))
        for i in range(mol.natm):
            for j in range(i):
                g = 1/atm_dist[i,j] * (grid_dist[i]-grid_dist[j])
                g = fadjust(i, j, g)
                g = (3 - g**2) * g * .5
                g = (3 - g**2) * g * .5
                g = (3 - g**2) * g * .5
                pbecke[i] *= .5 * (1-g + 1e-200)
                pbecke[j] *= .5 * (1+g + 1e-200)

        dpbecke = np.zeros((mol.natm,mol.natm,ngrids,3))
        for ia in range(mol.natm):
            for ib in range(mol.natm):
                if ib != ia:
                    g = 1/atm_dist[ia,ib] * (grid_dist[ia]-grid_dist[ib])
                    p0 = gadjust(ia, ib, g)
                    g = fadjust(ia, ib, g)
                    p1 = (3 - g **2) * g  * .5
                    p2 = (3 - p1**2) * p1 * .5
                    p3 = (3 - p2**2) * p2 * .5
                    s_uab = .5 * (1 - p3 + 1e-200)
                    t_uab = -27./16 * (1-p2**2) * (1-p1**2) * (1-g**2)
                    t_uab /= s_uab
                    t_uab *= p0

# * When grid is on atom ia/ib, ua/ub == 0, d_uba/d_uab may have huge error
#   How to remove this error?
# * JCP 98, 5612 (1993); (B8) (B10) miss many terms
                    uab = atm_coords[ia] - atm_coords[ib]
                    if ia == atom_id:  # dA PA: dA~ib, PA~ia
                        ua = atm_coords[ib] - coords
                        d_uab = ua/grid_dist[ib,:,None]/atm_dist[ia,ib]
                        v = (grid_dist[ia]-grid_dist[ib])/atm_dist[ia,ib]**3
                        d_uab-= v[:,None] * uab
                        dpbecke[ia,ia] += (pbecke[ia]*t_uab).reshape(-1,1) * d_uab
                    else:  # dB PB: dB~ib, PB~ia
                        ua = atm_coords[ia] - coords
                        d_uab = ua/grid_dist[ia,:,None]/atm_dist[ia,ib]
                        v = (grid_dist[ia]-grid_dist[ib])/atm_dist[ia,ib]**3
                        d_uab-= v[:,None] * uab
                        dpbecke[ia,ia] += (pbecke[ia]*t_uab).reshape(-1,1) * d_uab

                        if ib != atom_id:  # dA PB: dA~atom_id PB~ia D~ib
                            ua_ub = ((coords-atm_coords[ia])/grid_dist[ia,:,None] -
                                     (coords-atm_coords[ib])/grid_dist[ib,:,None])
                            ua_ub /= atm_dist[ia,ib]
                            dpbecke[atom_id,ia] += (pbecke[ia]*t_uab)[:,None] * ua_ub

                    uba = atm_coords[ib] - atm_coords[ia]
                    if ib == atom_id:  # dA PB: dA~ib PB~ia
                        ub = atm_coords[ia] - coords
                        d_uba = ub/grid_dist[ia,:,None]/atm_dist[ia,ib]
                        v = (grid_dist[ib]-grid_dist[ia])/atm_dist[ia,ib]**3
                        d_uba-= v[:,None] * uba
                        dpbecke[ib,ia] += -(pbecke[ia]*t_uab).reshape(-1,1) * d_uba
                    else:  # dB PC: dB~ib, PC~ia and dB PA: dB~ib, PA~ia
                        ub = atm_coords[ib] - coords
                        d_uba = ub/grid_dist[ib,:,None]/atm_dist[ia,ib]
                        v = (grid_dist[ib]-grid_dist[ia])/atm_dist[ia,ib]**3
                        d_uba-= v[:,None] * uba
                        dpbecke[ib,ia] += -(pbecke[ia]*t_uab).reshape(-1,1) * d_uba
        return pbecke, dpbecke

    ngrids = 0
    for ia in range(mol.natm):
        coords, vol = atom_grids_tab[mol.atom_symbol(ia)]
        ngrids += vol.size

    coords_all = np.zeros((ngrids,3))
    w0 = np.zeros((ngrids))
    w1 = np.zeros((mol.natm,ngrids,3))
    p1 = 0
    for ia in range(mol.natm):
        coords, vol = atom_grids_tab[mol.atom_symbol(ia)]
        coords = coords + atm_coords[ia]
        p0, p1 = p1, p1 + vol.size
        coords_all[p0:p1] = coords
        pbecke, dpbecke = gen_grid_partition(coords, ia)
        z = pbecke.sum(axis=0)
        for ib in range(mol.natm):  # derivative wrt to atom_ib
            dz = dpbecke[ib].sum(axis=0)
            w1[ib,p0:p1] = dpbecke[ib,ia]/z[:,None] - (pbecke[ia]/z**2)[:,None]*dz
            w1[ib,p0:p1] *= vol[:,None]

        w0[p0:p1] = vol * pbecke[ia] / z
    return coords_all, w0, w1

# JCP 98, 5612 (1993); DOI:10.1063/1.464906
def grids_response_cc(grids):
    mol = grids.mol
    atom_grids_tab = grids.gen_atomic_grids(mol, grids.atom_grid,
                                            grids.radi_method,
                                            grids.level, grids.prune)
    atm_coords = np.asarray(mol.atom_coords() , order='C')
    atm_dist = gto.inter_distance(mol, atm_coords)

    def _radii_adjust(mol, atomic_radii):
        charges = mol.atom_charges()
        if grids.radii_adjust == radi.treutler_atomic_radii_adjust:
            rad = np.sqrt(atomic_radii[charges]) + 1e-200
        elif grids.radii_adjust == radi.becke_atomic_radii_adjust:
            rad = atomic_radii[charges] + 1e-200
        else:
            fadjust = lambda i, j, g: g
            gadjust = lambda *args: 1
            return fadjust, gadjust

        rr = rad.reshape(-1,1) * (1./rad)
        a = .25 * (rr.T - rr)
        a[a<-.5] = -.5
        a[a>0.5] = 0.5

        def fadjust(i, j, g):
            return g + a[i,j]*(1-g**2)

        #: d[g + a[i,j]*(1-g**2)] /dg = 1 - 2*a[i,j]*g
        def gadjust(i, j, g):
            return 1 - 2*a[i,j]*g
        return fadjust, gadjust

    fadjust, gadjust = _radii_adjust(mol, grids.atomic_radii)

    def gen_grid_partition(coords, atom_id):
        ngrids = coords.shape[0]
        grid_dist = []
        grid_norm_vec = []
        for ia in range(mol.natm):
            v = (atm_coords[ia] - coords).T
            normv = np.linalg.norm(v,axis=0) + 1e-200
            v /= normv
            grid_dist.append(normv)
            grid_norm_vec.append(v)

        def get_du(ia, ib):  # JCP 98, 5612 (1993); (B10)
            uab = atm_coords[ia] - atm_coords[ib]
            duab = 1./atm_dist[ia,ib] * grid_norm_vec[ia]
            duab-= uab[:,None]/atm_dist[ia,ib]**3 * (grid_dist[ia]-grid_dist[ib])
            return duab

        pbecke = np.ones((mol.natm,ngrids))
        dpbecke = np.zeros((mol.natm,mol.natm,3,ngrids))
        for ia in range(mol.natm):
            for ib in range(ia):
                g = 1/atm_dist[ia,ib] * (grid_dist[ia]-grid_dist[ib])
                p0 = fadjust(ia, ib, g)
                p1 = (3 - p0**2) * p0 * .5
                p2 = (3 - p1**2) * p1 * .5
                p3 = (3 - p2**2) * p2 * .5
                t_uab = 27./16 * (1-p2**2) * (1-p1**2) * (1-p0**2) * gadjust(ia, ib, g)

                s_uab = .5 * (1 - p3 + 1e-200)
                s_uba = .5 * (1 + p3 + 1e-200)
                pbecke[ia] *= s_uab
                pbecke[ib] *= s_uba
                pt_uab =-t_uab / s_uab
                pt_uba = t_uab / s_uba

# * When grid is on atom ia/ib, ua/ub == 0, d_uba/d_uab may have huge error
#   How to remove this error?
                duab = get_du(ia, ib)
                duba = get_du(ib, ia)
                if ia == atom_id:
                    dpbecke[ia,ia] += pt_uab * duba
                    dpbecke[ia,ib] += pt_uba * duba
                else:
                    dpbecke[ia,ia] += pt_uab * duab
                    dpbecke[ia,ib] += pt_uba * duab

                if ib == atom_id:
                    dpbecke[ib,ib] -= pt_uba * duab
                    dpbecke[ib,ia] -= pt_uab * duab
                else:
                    dpbecke[ib,ib] -= pt_uba * duba
                    dpbecke[ib,ia] -= pt_uab * duba

# * JCP 98, 5612 (1993); (B8) (B10) miss many terms
                if ia != atom_id and ib != atom_id:
                    ua_ub = grid_norm_vec[ia] - grid_norm_vec[ib]
                    ua_ub /= atm_dist[ia,ib]
                    dpbecke[atom_id,ia] -= pt_uab * ua_ub
                    dpbecke[atom_id,ib] -= pt_uba * ua_ub

        for ia in range(mol.natm):
            dpbecke[:,ia] *= pbecke[ia]

        return pbecke, dpbecke

    natm = mol.natm
    for ia in range(natm):
        coords, vol = atom_grids_tab[mol.atom_symbol(ia)]
        coords = coords + atm_coords[ia]
        pbecke, dpbecke = gen_grid_partition(coords, ia)
        z = 1./pbecke.sum(axis=0)
        w1 = dpbecke[:,ia] * z
        w1 -= pbecke[ia] * z**2 * dpbecke.sum(axis=1)
        w1 *= vol
        w0 = vol * pbecke[ia] * z
        yield coords, w0, w1
