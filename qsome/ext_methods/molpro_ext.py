#An object to run a WF calculation using molpro.
from pyscf import lib
from string import Template
import re
import numpy as np
import os
import subprocess
from shutil import copyfile


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
read,h01,type=h0,
$HMAT
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
read,h01,type=h0,
$HMAT
save,h01,7500.1,h0}

$METHOD
put,molden,$FNAME;   !save orbitals in molden format
''')


class MolproExt:

    def __init__(self, mol, method, ext_pot, core_ham, file_name, work_dir, scr_dir, nproc, pmem, save_orbs, save_density, hl_dict):

        self.mol = mol
        self.method = method
        self.ext_pot = ext_pot
        self.core_ham = core_ham
        self.h0_name = None
        self.filename = file_name
        self.work_dir = work_dir
        self.scr_dir = scr_dir
        self.nproc = nproc
        self.pmem = pmem
        self.save_orbs = save_orbs
        self.save_density = save_density
        self.__set_hl_method_settings(hl_dict)
        #self.generate_molpro_input()

    def __set_hl_method_settings(self, hl_dict):
        """Sets the object parameters based on the hl settings

        Parameters
        ----------
        hl_dict : dict
            A dictionary containing the hl specific settings.
        """

        if hl_dict is None:
            hl_dict = {}
        self.hl_dict = hl_dict
           
        if 'cc' in self.method:
            self.cc_loc_orbs = hl_dict.get("loc_orbs")
            self.cc_initguess = hl_dict.get("cc_initguess")
            self.cc_froz_core_orbs = hl_dict.get("froz_core_orbs")
        if 'cas' in self.method:
            self.cas_loc_orbs = hl_dict.get("loc_orbs")
            self.cas_init_guess = hl_dict.get("cas_initguess")
            self.cas_active_orbs = hl_dict.get("active_orbs")
            self.cas_avas = hl_dict.get("avas")


    def generate_molpro_input(self):
  
        #Need to figure out the H0
        nao = self.mol.nao_nr()
        emb_ham = self.core_ham + (self.ext_pot[0] + self.ext_pot[1])/2.
        #print (self.ext_pot[0])
        #print (self.ext_pot[1])
        I_PyscfMolpro = ShPyscf2Molpro(self.mol)
        emb_ham = emb_ham[I_PyscfMolpro, :]
        emb_ham = emb_ham[: ,I_PyscfMolpro]
        h0 = emb_ham.reshape(nao,nao)
        h0_string = "BEGIN_DATA,\n"
        h0_string += "# MATRIX H0                 H0    SYMMETRY=1\n"
        for x in range(nao):
           i=0
           for y in h0[x]:
               i += 1
               h0_string += "%15.8f," % y
               if i % 5 == 0: 
                   h0_string += "\n"
           if nao % 5 != 0:
               h0_string += "\n"
        h0_string += "END_DATA,\n"


        method_string = ""
        if self.method in ['hf', 'hartree fock', 'hartree-fock']:
            method_string = '{hf;noenest}'
        elif self.method == 'ccsd':
            method_string = '{hf;noenest}\n'
            method_string += '{rccsd;core'
            if self.cc_froz_core_orbs:
                method_string += ',' + str(self.cc_froz_core_orbs)
            method_string += '}'
        elif self.method == 'ccsd(t)':
            method_string = '{hf;noenest}\n'
            method_string += '{rccsd(t);core'
            if self.cc_froz_core_orbs:
                method_string += ',' + str(self.cc_froz_core_orbs)
            method_string += '}'
        elif self.method == 'fcidump':
            dump_filename = 'FCIDUMP'
            method_string = '{hf;noenest}\n'
            method_string += '''{fci,dump=''' + dump_filename + ''';core;}\n'''
        elif self.method == 'rohf/rccsd(t)':
            method_string = '{rhf;noenest;\n'
            num_elec = self.mol.tot_electrons()
            method_string += 'wf,' + str(num_elec) + ',1,' + str(np.abs(self.mol.spin)) + ';}\n'
            method_string += '{rccsd(t);core'
            if self.cc_froz_core_orbs:
                method_string += ',' + str(self.cc_froz_core_orbs)
            method_string += '}'
        elif re.match(re.compile('cas(pt2)?\[.*\].*'), self.method):
            cas_space = [int(i) for i in (self.method[self.method.find("[") + 1:self.method.find("]")]).split(',')]
            num_elec = self.mol.tot_electrons()
            num_closed = int((num_elec - cas_space[0]) / 2.)
            num_occ = num_closed + cas_space[1]
            if self.cas_avas is None:
                if self.mol.spin != 0:
                    method_string = f'{{uhf,maxit=240;noenest;wf,{num_elec},1,{np.abs(self.mol.spin)}}}\n'
                else:
                    method_string = '{hf;noenest;}\n'
            else:
                if self.mol.spin != 0:
                    method_string = '{uhft;avas,open=1;' + ';'.join(self.cas_avas) + '}\n'
                else:
                    method_string = '{rhft;avas;' + ';'.join(self.cas_avas) + '}\n'
            method_string += "put,molden," + os.path.splitext(self.filename.split('/')[-1])[0] + "_hf.molden;   !save orbitals in molden format\n"
            method_string += "{casscf\n"
            method_string += "maxiter,100\n"
            method_string += "closed," + str(num_closed) + "\n"
            method_string += "occ," + str(num_occ) + "\n"
            method_string += "wf," + str(num_elec) + ",1," + str(np.abs(self.mol.spin)) + "\n"

            if self.cas_active_orbs:
                occ_orbs = [str(x+.1) for x in self.cas_active_orbs if x <= int(num_elec/2)]
                vir_orbs = [str(x+.1) for x in self.cas_active_orbs if x > int(num_elec/2)]
                for i in range(int(cas_space[0]/2)):
                    method_string += "rotate," + str(int(int(num_elec/2) - i)+.1) + "," + occ_orbs[-i] + ",0;\n"

                for j in range(len(vir_orbs)):
                    method_string += "rotate," + str(int(int(num_elec/2) + j + 1)+.1) + "," + vir_orbs[j] + ",0;\n"

            method_string += "}"

            if re.match(re.compile('caspt2\[.*\]'), self.method):
                method_string += "\n{rs2c,maxiti=100;core}"

            if ('nevpt2' in self.method.lower()):
                method_string += '\nnevpt2'

        else:
            pass

        mol_geom = self.pyscf2molpro_geom()
        mol_basis = self.pyscf2molpro_basis()
        dummy_atoms = ""
        #for gh_num in range(len(self.mol.ghosts)):
        #    dummy_atoms += self.mol.ghosts[gh_num] + str(gh_num + 1) + ","
        ##Remove final comma.
        #dummy_atoms = dummy_atoms[:-1]

        pword = self.pmem / 8.
        orb_file = os.path.splitext(self.filename.split('/')[-1])[0] + ".molden"
        inp_str = molpro_template_molden.substitute(MEMORY=str(pword), SYMMETRY="nosym",
                      BASIS=mol_basis, GEOM=mol_geom, DUMMY=dummy_atoms, CHARGE=self.mol.charge,
                      SPIN=self.mol.spin, HMAT=h0_string, METHOD=method_string, FNAME=orb_file) 
        return inp_str

    def pyscf2molpro_geom(self):
        data = [[0 for x in range(4)] for y in range(self.mol.natm)]
        atom_geom = [0 for x in range(self.mol.natm)]
        gh_num = 0
        for ia in range(self.mol.natm):
            symb = self.mol.atom_pure_symbol(ia)
            if (symb.upper() == "GHOST"):
                symb = self.mol.ghosts[gh_num] + str(gh_num + 1)
                gh_num += 1
            coord = self.mol.atom_coord(ia)
            coord[:] = [ x * lib.param.BOHR for x in coord ]
            data[ia][0]=symb
            data[ia][1]=str(coord[0])
            data[ia][2]=str(coord[1])
            data[ia][3]=str(coord[2])
            atom_geom[ia] = "  ".join(data[ia])
        geom = '\n'.join(atom_geom)
        return geom

    def pyscf2molpro_basis(self):
        basis_string = '{\n'
        for basis_symb in self.mol._basis.keys():
            if (basis_symb.split(":")[0].upper() == 'GHOST'):
                ghost_basis = self.mol._basis[basis_symb]
                basis_symb = mol.ghosts[int(basis_symb.split(':')[1])-1] + basis_symb.split(":")[1]
                basis_string += convert_basis_to_molpro(basis_symb, ghost_basis)
            else:
                basis_string += convert_basis_to_molpro(basis_symb, self.mol._basis[basis_symb])
            #if ('hf' not in self.method) and ('hartree-fock' not in self.method):
            #    basis_string += '\nset,mp2fit\ndefault, %s/mp2fit'%basis
        basis_string += '}'
        return basis_string


    def get_energy(self):
        input_file = self.generate_molpro_input()
        with open (self.work_dir + '/' + self.filename + '_molpro.com','w') as f:
            f.write(input_file)
        ##Run molpro
        file_path = self.work_dir + '/' + self.filename + '_molpro.com'
        print (file_path)
        cmd = ' '.join(("molpro" + ' -n ' + str(self.nproc) + ' -d ' + self.scr_dir, file_path))
        print (cmd)
        proc_results = subprocess.getoutput(cmd)
        print (proc_results)

        #Copy the molden files if caspt2
        if re.match(re.compile('cas(pt2)?\[.*\].*'), self.method):
            curr_work_dir = os.getcwd()
            hf_orb_file = curr_work_dir + "/" + os.path.splitext(self.filename)[0].lower() + "_hf.molden" 
            cas_orb_file = curr_work_dir + "/" + os.path.splitext(self.filename)[0].lower() + ".molden" 
            copyfile(hf_orb_file, self.work_dir + '/' + 
                     os.path.splitext(self.filename)[0] + '_hf.molden')
            copyfile(cas_orb_file, self.work_dir + '/' + 
                     os.path.splitext(self.filename)[0] + '_cas.molden')

        #Open and extract from output.
        outfile = self.work_dir + '/' + self.filename + '_molpro.out'
        with open(outfile, 'r') as fin:
            dat = fin.read()

            #Copy fcidump to the correct location.
            if self.method == 'fcidump':
                #fcidump_loc = dat.find('Transformed integrals will be written to file')
                #fcidump_end = dat.find('FCIDUMP', fcidump_loc)
                #fcidump_path = dat[fcidump_loc:fcidump_end].split()[-1] + 'FCIDUMP'
                curr_num = -1
                curr_path = self.scr_dir
                while os.path.exists(curr_path):
                    curr_num += 1
                    curr_path = curr_path.rsplit('-', 1)[0]
                    curr_path += '-' + str(curr_num)
                path_num = curr_num - 1
                path_name = self.scr_dir
                if path_num >= 0:
                    path_name = path_name + '-' + str(path_num)
                fcidump_path = path_name + '/fcidump'
                print (fcidump_path)
                copyfile(fcidump_path, self.work_dir + '/' + self.filename + '.fcidump')
                os.remove(fcidump_path)

            dat1 = dat.splitlines()
            nums = []
            elec_e = []
            if self.method == 'hf' or self.method == 'hartree-fock' or self.method == 'fcidump':
                for i in range((len(dat1)-10), len(dat1), 1):
                    if "HF-SCF" in dat1[i]:
                        elec_e = dat1[i+1].split()

            if self.method == 'ccsd' or self.method == 'ccsd(t)' or self.method == 'rohf/rccsd(t)':
                for i in range((len(dat1)-10), len(dat1), 1):
                    if "CCSD" in dat1[i]:
                        elec_e = dat1[i+1].split()

            if re.match(re.compile('cas(pt2)?\[.*\].*'), self.method):
                for i in range((len(dat1)-10), len(dat1), 1):
                    if (self.cas_avas is None and "HF-SCF" in dat1[i]) or ("RHFT" in dat1[i]):
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
        energy = [(float(i) + nuc_e) for i in elec_e]
        return energy
