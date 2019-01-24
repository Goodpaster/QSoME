#!/usr/bin/env python

import os, sys
import shutil
import tempfile
import getpass
import subprocess  
import re
from string import Template

from os.path import expanduser, expandvars, abspath, splitext, dirname

import numpy as np

from pyscf import gto, scf, lib, ao2mo
from pyscf.tools import fcidump

from shutil import copyfile, move

import multiprocessing
import psutil

#TODO: Changes some part of the pyscf object such that if get_interaction_energy is called after molpro_energy, it returns incorrect results.

#MOLPROEXE = os.environ['HOME'] + '/workspace/molpro-dev/bin/molpro'
#nprocs = multiprocessing.cpu_count()
#mempp = 450
#MOLPROEXE = 'molpro -n 8'#+str(nprocs)

molpro_template=Template('''
!leave this line blank
memory, $MEMORY, m
symmetry, $SYMMETRY
basis=$BASIS
geometry={$GEOM}
dummy,$DUMMY
charge=$CHARGE
spin=$SPIN

{matrop                   !read the modified core hamiltonian
read,h01,type=h0,file=$HMAT
save,h01,7500.1,h0}

$METHOD
''')

molpro_template_molden=Template('''
!leave this line blank
memory, $MEMORY, m
symmetry, $SYMMETRY
basis=$BASIS
geometry={$GEOM}
dummy,$DUMMY
charge=$CHARGE
spin=$SPIN

{matrop                   !read the modified core hamiltonian
read,h01,file=$HMAT
save,h01,1210.1,h0}

$METHOD
put,molden,$FNAME;   !save orbitals in molden format
''')

def molpro_energy(mol, h0, method, in_file='temp', output_orbs=False, active_orbs=None, localize=False, work_dir=None, scr_dir=None, nproc=None, pmem=None):

    ## Prepare input files
    if nproc is None:
        nproc = 1
    if pmem is None:
        pmem = 2000
    pmem = pmem - 10
    pword = pmem / 8.0
    username = getpass.getuser()
    if scr_dir is None:
        scr_dir = '/scratch.local/'+username+'/temp_molpro'
    if not os.path.exists(scr_dir):
        os.makedirs(scr_dir)

    if work_dir is None:
        work_dir = dirname(abspath(expandvars(expanduser(in_file))))
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    

    input_file = os.path.splitext(os.path.split(in_file)[-1])[0] + "_molpro.com"
    input_file_scr = os.path.join((scr_dir + '/'), (os.path.splitext(os.path.split(input_file)[1])[0] + '.com'))

    I_PyscfMolpro = ShPyscf2Molpro(mol)
    h0 = h0[I_PyscfMolpro, :]
    h0 = h0[: ,I_PyscfMolpro]
    h0_name = os.path.splitext(os.path.split(in_file)[-1])[0][-2:] + "_h0.mat"
    h0_scr = (scr_dir + "/" + h0_name)

    file_mod = 1
    base_name = h0_name
    while (os.path.isfile(h0_scr)):
        h0_name = str(file_mod) + "_" + base_name[2:]
        h0_scr = scr_dir + "/" + h0_name
        file_mod += 1

    write_h0(h0_scr, mol, h0)
    molpro_input = generate_molpro_input(mol, method, work_dir, input_file, h0_name, output_orbs, active_orbs, localize, pword)

    with open(work_dir + '/' + input_file, 'w') as f:
        f.write(molpro_input)

    with open(scr_dir + '/' + input_file, 'w') as f:
        f.write(molpro_input)

    ## Run MolPro with generated input files
    cmd = ' '.join(('cd ' + scr_dir + " && molpro" + ' -n ' + str(nproc) + ' -d ' + scr_dir, input_file_scr))
    proc_results = subprocess.getoutput(cmd) 
    print (proc_results)
    outfile = scr_dir + '/' + os.path.splitext(input_file)[0] + ".out"

    copyfile(h0_scr, (work_dir + '/' + os.path.splitext(input_file)[0] + '.mat'))
    copyfile(outfile, (work_dir + '/' + os.path.splitext(input_file)[0] + '.out'))
    if output_orbs:
        orbfile = scr_dir + "/" + os.path.splitext(input_file)[0].lower() + ".molden"
        orbfile2 = scr_dir + "/" + os.path.splitext(input_file)[0].lower() + "_hf.molden"
        copyfile(orbfile, (work_dir + '/' + os.path.splitext(input_file)[0] + '.molden'))
        if re.match(re.compile('cas(pt2)?\[.*\].*'), method):
            copyfile(orbfile2, (work_dir + '/' + os.path.splitext(input_file)[0] + '_hf.molden'))
    if method == 'fcidump':
        FCIDUMP = scr_dir + "/" + "FCIDUMP"
        copyfile(FCIDUMP, (os.path.splitext(input_file)[0] + '.fcidump'))
        print (f"FCIDUMP ORIGINAL FILE: {FCIDUMP}")
        print (f"FCIDUMP file location: {os.path.splitext(input_file)[0] + '.fcidump'}")
    if 'fehler' in proc_results or 'improper' in proc_results:
        sys.stderr.write('molpro tempfiles in %s\n'%scr_dir)
        raise RuntimeError('molpro fail as:\n' + proc_results)
    
    # Extract data from MolPro output file
    with open(outfile, 'r') as fin:

        dat = fin.read()
        dat1 = dat.splitlines()
        nums = []
        elec_e = []
        if method == 'hf' or method == 'hartree-fock' or method == 'fcidump':
            for i in range((len(dat1)-10), len(dat1), 1):
                if "HF-SCF" in dat1[i]:
                    elec_e = dat1[i+1].split()

        if method == 'ccsd' or method == 'ccsd(t)':
            for i in range((len(dat1)-10), len(dat1), 1):
                if "CCSD" in dat1[i]:
                    elec_e = dat1[i+1].split()

        if re.match(re.compile('cas(pt2)?\[.*\].*'), method):
            for i in range((len(dat1)-10), len(dat1), 1):
                if "HF-SCF" in dat1[i]:
                    elec_e = dat1[i+1].split()

        else:
            for num, line in enumerate(dat1):
                if 'Results for state' in line:
                    energies.append(line)
                if 'oscill.stren.' in line: 
                    nums.append(num)
            if nums:
                energies = [dat1[i+1] for i in nums]

    found_nuc = False
    for j in range(len(dat1)):
        if "NUCLEAR REPULSION ENERGY" in dat1[j]:
            nuc_e = float(dat1[j].split()[3])
            found_nuc = True
            break
    if not found_nuc:
        #Coordinates  must be converted to Bohr.
        nuc_e = mf.mol.energy_nuc(coords=(mf.mol.atom_coords() * 1.889725989))

    energy = [(float(i) + nuc_e) for i in elec_e]
    return energy
    

def generate_molpro_input(mol, method, work_dir, input_file, h0_filename, output_orbs, active_orbs, localize, pword):
    
    if (method == 'hf') or (method == 'hartree fock') or (method == 'hartree-fock'):
        method_string = '{hf;noenest}'

    if re.match(re.compile('cas(pt2)?\[.*\].*'), method):
        cas_space = [int(i) for i in (method[method.find("[") + 1:method.find("]")]).split(',')]
        num_elec = mol.tot_electrons()
        num_closed = int((num_elec - cas_space[0]) / 2)
        num_occ = num_closed + cas_space[1]
        method_string = '{hf;noenest}\n'
        if localize:
            method_string += 'locali\n'

        if output_orbs:
            method_string += "put,molden," + os.path.splitext(input_file)[0] + "_hf.molden;   !save orbitals in molden format\n"
        method_string += "{casscf\n"
        method_string += "closed," + str(num_closed) + "\n"
        method_string += "occ," + str(num_occ) + "\n"
        method_string += "wf," + str(num_elec) + ",1," + str(mol.spin) + "\n"

        if active_orbs:
            occ_orbs = [str(x+.1) for x in active_orbs if x <= int(num_elec/2)]
            vir_orbs = [str(x+.1) for x in active_orbs if x > int(num_elec/2)]
            for i in range(int(cas_space[0]/2)):
                method_string += "rotate," + str(int(int(num_elec/2) - i)+.1) + "," + occ_orbs[-i] + ",0;\n"

            for j in range(len(vir_orbs)):
                method_string += "rotate," + str(int(int(num_elec/2) + j + 1)+.1) + "," + vir_orbs[j] + ",0;\n"
                
        method_string += "}"

        if re.match(re.compile('caspt2\[.*\]'), method):
            method_string += "\nrs2c"

        if ('nevpt2' in method.lower()):
            method_string += '\nnevpt2'

        if ('mrcic' in method.lower()):
            method_string += '\nmrcic'

    if (method == 'ccsd'):
        method_string = '{hf;noenest}\n'
        if mol.spin != 0:
            method_string += 'uccsd'
        else:
            method_string += 'ccsd'
    if (method == 'ccsd(t)'):
        method_string = '{hf;noenest}\n'
        if mol.spin != 0:
            method_string += 'uccsd(t)'
        else:
            method_string += 'ccsd(t)'
    if (method == 'fcidump'):
        method_string = '{rhf;noenest}\n{fci;core;dump}\n'

    

    orb_file = os.path.splitext(input_file)[0] + ".molden"

    mol_geom = pyscf2molpro_geom(mol) 
    mol_basis = pyscf2molpro_basis(mol, method_string)
    dummy_atoms = ""
    for gh_num in range(len(mol.ghosts)):
        dummy_atoms += mol.ghosts[gh_num] + str(gh_num + 1) + ","

    #Remove final comma.
    dummy_atoms = dummy_atoms[:-1]
    if output_orbs:
        inp_str = molpro_template_molden.substitute(MEMORY=str(pword), SYMMETRY="nosym",
                                  BASIS=mol_basis, GEOM=mol_geom, DUMMY=dummy_atoms, CHARGE=mol.charge,
                                  SPIN=mol.spin, HMAT=h0_filename.split('/')[-1], METHOD=method_string, FNAME=orb_file) 
    else:
        inp_str = molpro_template.substitute(MEMORY=str(pword), SYMMETRY="nosym",
                                  BASIS=mol_basis, GEOM=mol_geom, DUMMY=dummy_atoms, CHARGE=mol.charge,
                                  SPIN=mol.spin, HMAT=h0_filename.split('/')[-1], METHOD=method_string) 

    return inp_str

def convert_basis_to_molpro(symb, basis):
    '''Convert the internal basis format to Molpro format string'''
    from pyscf.gto.mole import _std_symbol
    SPDF = ('S', 'P', 'D', 'F', 'G', 'H', 'I', 'J')
    MAXL = 8
    MAPSPDF = {'S': 0,
               'P': 1,
               'D': 2,
               'F': 3,
               'G': 4,
               'H': 5,
               'I': 6,
               'J': 7}
    res = []
    #symb = _std_symbol(symb)
    #print('basis',basis)

    # pass 1: comment line
    ls = [b[0] for b in basis]
    nprims = [len(b[1:]) for b in basis]
    nctrs = [len(b[1])-1 for b in basis]
    prim_to_ctr = {}
    for i, l in enumerate(ls):
        if l in prim_to_ctr:
            prim_to_ctr[l][0] += nprims[i]
            prim_to_ctr[l][1] += nctrs[i]
        else:
            prim_to_ctr[l] = [nprims[i], nctrs[i]]
    nprims = []
    nctrs = []
    for l in set(ls):
        nprims.append(str(prim_to_ctr[l][0])+SPDF[l].lower())
        nctrs.append(str(prim_to_ctr[l][1])+SPDF[l].lower())
    res.append('\n!' )
    res.append('! %s        (%s) -> [%s]' % (symb, ','.join(nprims), ','.join(nctrs)))
    res.append('! %s        (%s) -> [%s]' % (symb, ','.join(nprims), ','.join(nctrs)))

    # pass 2: basis data
    max_shell = max(ls) + 1
    exp = [[] for i in range(max_shell)]
    coeff = [[] for i in range(max_shell)]
    # create exponent list for each shell
    for bas in basis: 
        for j in range(max_shell):
            if ( bas[0] == j ): 
                exp[j].append(list(dat[0] for dat in bas[1:]))
                coeff[j].append(list(dat[1:] for dat in bas[1:]))
    #print('coeff',coeff)
    exp_flat = exp[:]
    for l in set(ls):
        exp_flat[l] = sum(exp_flat[l], [])
        res.append('%s,  %s ,  %s' % (SPDF[l].lower(), symb.upper(), '  ,'.join(str(s) for s in exp_flat[l])))
        start = 1
        end = 0
        for m in range(len(coeff[l])):
            if m == 0:
                end += len(coeff[l][m])
                #print('coeff[l][m]',coeff[l][m])
                for n in range(len(coeff[l][m][0])):
                    #print('len(coeff[l][m]',len(coeff[l][m]))
                    res.append('c,  %d.%d,  %s ' % (start,end, ',  '.join(str(s[n]) for s in coeff[l][m])))
            else:
                start += len(coeff[l][m-1])
                end += len(coeff[l][m])
                for n in range(len(coeff[l][m][0])):
                    res.append('c,  %d.%d,  %s ' % (start,end, ',  '.join(str(s[n]) for s in coeff[l][m])))

    return '\n'.join(res)

def pyscf2molpro_basis(mol, method):
    basis_string = '{\n'
    for basis_symb in mol._basis.keys():
        if (basis_symb.split(":")[0].upper() == 'GHOST'):
            ghost_basis = mol._basis[basis_symb]
            basis_symb = mol.ghosts[int(basis_symb.split(':')[1])-1] + basis_symb.split(":")[1]
            basis_string += convert_basis_to_molpro(basis_symb, ghost_basis)
        else:
            basis_string += convert_basis_to_molpro(basis_symb, mol._basis[basis_symb])
        if ('hf' not in method) and ('hartree-fock' not in method):
            basis_string += '\nset,mp2fit\ndefault, %s/mp2fit'%basis
    basis_string += '}'
    return basis_string
      
#Convert pyscf geometry to molpro format 
def pyscf2molpro_geom(mol):
    data = [[0 for x in range(4)] for y in range(mol.natm)]
    atom_geom = [0 for x in range(mol.natm)]
    gh_num = 0
    for ia in range(mol.natm):
        symb = mol.atom_pure_symbol(ia)
        if (symb.upper() == "GHOST"):
            symb = mol.ghosts[gh_num] + str(gh_num + 1)
            gh_num += 1
        coord = mol.atom_coord(ia)
        coord[:] = [ x * lib.param.BOHR for x in coord ]
        data[ia][0]=symb
        data[ia][1]=str(coord[0])
        data[ia][2]=str(coord[1])
        data[ia][3]=str(coord[2])
        atom_geom[ia] = "  ".join(data[ia])
    geom = '\n'.join(atom_geom)
    return geom 

#Write the new core Hamiltonian file
def write_h0(fname, mol, h0):
    nao = mol.nao_nr()
    h0 = h0.reshape(nao,nao)
    with open(fname, 'w') as fin:
        fin.write('BEGIN_DATA,\n')
        fin.write('# MATRIX H0                 H0    SYMMETRY=1\n')
        for x in range(nao):
            i=0
            for y in h0[x]:
                i += 1
                fin.write('%15.8f,' % y)
                if i % 5 == 0: 
                    fin.write('\n') 
            if nao % 5 != 0:
                fin.write('\n')
        fin.write('END_DATA,\n')

_Molpro2PyscfBasisPermSph = {
   0: [0],
   1: [0, 1, 2],
   #  0  1   2   3   4
   # D0 D-2 D+1 D+2 D-1
   2: [1, 4, 0, 2, 3],
   #  0   1   2  3   4   5   6
   # F+1 F-1 F0 F+3 F-2 F-3 F+2
   3: [5, 4, 1, 2, 0, 6, 3],
   #  0   1   2   3   4   5   6   7   8
   # G0 G-2 G+1 G+4 G-1 G+2 G-4 G+3 G-3
   4: [6, 8, 1, 4, 0, 2, 5, 7, 3],
   #  0   1   2   3   4   5   6   7   8  9   10
   # H+1 H-1 H+2 H+3 H-4 H-3 H+4 H-5 H0 H+5 H-2
   5: [7, 4, 5, 10, 1, 8, 0, 2, 3, 6, 9],
   #  0   1   2   3   4   5   6   7   8   9  10  11  12
   # I+6 I-2 I+5 I+4 I-5 I+2 I-6 I+3 I-4 I0 I-3 I-1 I+1
   6: [6, 4, 8, 10, 1, 11, 9, 12, 5, 7, 3, 2, 0],
}

_Pyscf2MolproBasisPermSph = {
   0: [0],
   1: [0, 1, 2],
   #  0  1   2   3   4
   # D0 D-2 D+1 D+2 D-1
   2: [2, 0, 3, 4, 1],
   #  0   1   2  3   4   5   6
   # F+1 F-1 F0 F+3 F-2 F-3 F+2
   3: [4, 2, 3, 6, 1, 0, 5],
   #  0   1   2   3   4   5   6   7   8
   # G0 G-2 G+1 G+4 G-1 G+2 G-4 G+3 G-3
   4: [4, 2, 5, 8, 3, 6, 0, 7, 1],
   #  0   1   2   3   4   5   6   7   8  9   10
   # H+1 H-1 H+2 H+3 H-4 H-3 H+4 H-5 H0 H+5 H-2
   5: [6, 4, 7, 8, 1, 2, 9, 0, 5, 10, 3],
   #  0   1   2   3   4   5   6   7   8   9  10  11  12
   # I+6 I-2 I+5 I+4 I-5 I+2 I-6 I+3 I-4 I0 I-3 I-1 I+1
   6: [12, 4, 11, 10, 1, 8, 0, 9, 2, 6, 3, 5, 7],
}

def ShPyscf2Molpro(mol):
    ###################################################################
    ###   create a index list with the size of AO basis 9/13/17     ###
    ###################################################################
    I_PyscfToMolpro = []
    iOff = 0
    # Must be the total atoms, not the basis keys.
    symbol_list = []
    ghost = 1
    for ia in range(mol.natm):
       
        symb = mol.atom_pure_symbol(ia)
        if symb == 'Ghost':
            symb = symb + ':' + str(ghost)
            ghost += 1
        symbol_list.append(symb)

    for basis_symb in symbol_list:
        index = []
        # pass 1: comment line
        ls = [bs[0] for bs in mol._basis[basis_symb]]
        nprims = [len(bs[1:]) for bs in mol._basis[basis_symb]]
        nctrs = [len(bs[1])-1 for bs in mol._basis[basis_symb]]
        prim_to_ctr = {}
        for i, l in enumerate(ls):
            if l in prim_to_ctr:
                prim_to_ctr[l][0] += nprims[i]
                prim_to_ctr[l][1] += nctrs[i]
            else:
                prim_to_ctr[l] = [nprims[i], nctrs[i]]
        for l in set(ls): 
            for i in range(prim_to_ctr[l][1]): 
                index.append(l)
        for l in index:
            I_PyscfToMolpro += [(o + iOff) for o in _Pyscf2MolproBasisPermSph[l]] 
            iOff += 2*l + 1
    I_PyscfToMolpro = np.array(I_PyscfToMolpro)  
    return I_PyscfToMolpro

if __name__ == '__main__':
    import os, sys
    import numpy
    from pyscf import gto
    from pyscf import scf,dft

    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [[ 'F', [1.1,0.0,0.0]],
                [ 'H', [0.0, 0.0, 0.0]]] 
    mol.basis = '321g'
    mol.build()
    mf = scf.RHF(mol)
    print(mf.kernel())
    print('PYSCF HF',mf.e_tot )
    print('PYSCF MO_ENERGIES',mf.mo_energy)

    e = molpro_energy(mf, 'cas[2,2]/nevpt2', mf.get_hcore(), '321g')
    print('MOLPRO HF', e) # -4.28524318
    print('------------')
