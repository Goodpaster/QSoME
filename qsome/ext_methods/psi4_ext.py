# interface to run psi4 from the embedded system.
# modified Psi4 need to be complied from source to read embedding potential
# compiled version can be found in /home/goodpast/shared/Xuelan/psi4.
# change made to psi4/psi4/src/psi4/libscf_solver/hf.cc to read embpot.dat.
# add the following PSI4 executable2 to your ~/.bashrc:
# export PATH=/home/goodpast/shared/Xuelan/psi4/psi4/objdir/stage/bin:$PATH
# export PSI_SCRATCH=/scratch.global/${yourusername}/

from string import Template
import numpy as np
import os
import subprocess


class Psi4Ext:

    def __init__(self,  mol,  method, ext_pot, core_ham, filename, work_dir, scr_dir, nproc, pmem, nroots, root):
        self.mol = mol            # mol object passed from PySCF
        self.method = method      # excited-state method name
        self.ext_pot = ext_pot    # external potential/embedding potential
        self.core_ham = core_ham  # core_hamiltonian (unmodified)
        self.filename = filename
        self.work_dir = work_dir
        self.scr_dir = scr_dir
        self.nproc  = nproc
        self.pmem = pmem          # in GB
        self.nroots = nroots      # # of roots for EOM-CCSD (suggest > root)
        self.root = root          # the root to be calculated by eom-cc3 

    def generate_psi4_input(self):
        if self.nproc is not None and self.pmem is not None:
            memory = self.pmem * self.nproc 
        else:
            memory = 20
        nao = self.mol.nao_nr()
        mol_geom = self.pyscf2psi4_geom()
        mol_basis = self.pyscf2psi4_basis()
        if self.mol.cart == True:
            basis_type = "cartesian"
        else:
            basis_type = "spherical"
        # mol.spin = 2S == nelec_alpha - nelec_beta
        # SPIN = 2S+ 1
        inp_str = psi4_eomcc_template.substitute(MEMORY=memory,
                  BASIS=mol_basis, GEOM=mol_geom, CHARGE=self.mol.charge,
                  SPIN=self.mol.spin+1, METHOD=self.method, NROOT=self.nroots,
                  BASIS_TYPE=basis_type, NCORE=self.nproc)
        return inp_str

    def generate_psi4_embpot(self, emb_ham):
        nao = self.mol.nao_nr()
        #emb_ham = self.core_ham + self.ext_pot
        f = open('embpot.dat', 'w')
        for i in range(nao):
            for j in range(nao):
                f.write("%15.12f\n" % (emb_ham)[i, j])
        f.close()

    def get_energy(self):
        # generate PSI4 input file
        input_file = self.generate_psi4_input()
        with open(self.work_dir + '/' + self.filename + '_psi4.dat', 'w') as f:
            f.write(input_file)
        f.close()

        # generate embpot.dat, core_Hamiltonian should be included in embpot
        emb_ham = self.core_ham + (self.ext_pot[0] + self.ext_pot[1])/2.
        nao = self.mol.nao_nr()

        ## debug: print pyscf embpot
        #h0 =  emb_ham.reshape(nao,nao)
        #h0_string = ''
        #for x in range(nao):
        #   i=0
        #   for y in h0[x]:
        #       i += 1
        #       h0_string += "%15.8f" % y
        #       if i % 5 == 0: 
        #           h0_string += "\n"
        #   if nao % 5 != 0:
        #       h0_string += "\n"
        #with open(self.work_dir + '/embpot_pyscf.dat', 'w') as f:
        #    print('embpot_pyscf.dat is saved to {}'.format(f))
        #    f.write(h0_string)
        #f.close()


        I_Pyscf2Psi4 = ShPyscf2Psi4(self.mol)
        emb_ham = emb_ham[I_Pyscf2Psi4, :]
        emb_ham = emb_ham[: ,I_Pyscf2Psi4]
        nao = self.mol.nao_nr()
        h0 = emb_ham.reshape(nao,nao)
        self.generate_psi4_embpot(h0)
    
        # Run psi4
        file_path = self.work_dir + '/' + self.filename + '_psi4.dat'
        #print('embpot.dat is saved to {}'.format(file_path))
        cmd = ' '.join(["psi4 -n", str(self.nproc),file_path])
        #print(cmd)
        proc_results = subprocess.getoutput(cmd)
        print(proc_results)

        
        # Open and extract from output.
        energy = []
        outfile = self.work_dir + '/' + self.filename + '_psi4.out'
        with open(outfile, 'r') as fin:
            dat = fin.read()
            dat1 = dat.splitlines()
            # Copy fcidump to the correct location.
            for num, line in enumerate(dat1):
                if 'RHF Final Energy' in line:
                    energy.insert(0, float(line.split()[-1]))
                if 'CCSD total energy' in line:
                    energy.insert(0, float(line.split()[-1]))
                if 'CC3 total energy' in line:
                    energy.insert(0, float(line.split()[-1]))
                if 'Completed EOM_CCSD' in line:
                    # only insert the value for desird root
                    energy.insert(0,float(dat1[num-self.nroots+self.root].split()[1]))
                    for i in range(self.nroots):
                        e_eomccsd = dat1[num-self.nroots+self.root].split()[1]
                        print('EOM-CCSD for root {} is {} eV'.format(i+1, e_eomccsd))
                if 'EOM State 1' in line:
                    energy.insert(0, float(line.split()[3]))

        
        return energy


    def pyscf2psi4_geom(self):
        # set unit to angstrom, which is default in Psi4
        data = [[0 for x in range(4)] for y in range(self.mol.natm)]
        atom_geom = [0 for x in range(self.mol.natm)]
        gh_num = 0
        for ia in range(self.mol.natm):
            symb = self.mol.atom_pure_symbol(ia)
            if (symb.upper() == "GHOST"):
                symb = self.mol.ghosts[gh_num] + str(gh_num + 1)
                gh_num += 1
            coord = self.mol.atom_coord(ia, unit='ANG')
            data[ia][0] = symb
            data[ia][1] = str(coord[0])
            data[ia][2] = str(coord[1])
            data[ia][3] = str(coord[2])
            atom_geom[ia] = "  ".join(data[ia])
        geom = '\n'.join(atom_geom)
        return geom
    
    
    def pyscf2psi4_basis(self):
        # combine basis set for different atoms
        basis_string = '****\n'
        for basis_symb in self.mol._basis.keys():
            if (basis_symb.split(":")[0].upper() == 'GHOST'):
                ghost_basis = self.mol._basis[basis_symb]
                basis_symb = mol.ghosts[int(basis_symb.split(
                    ':')[1])-1] + basis_symb.split(":")[1]
                basis_string += convert_basis_to_psi4(basis_symb, ghost_basis)
            else:
                basis_string += convert_basis_to_psi4(
                    basis_symb, self.mol._basis[basis_symb])
            # if ('hf' not in self.method) and ('hartree-fock' not in self.method):
            #    basis_string += '\nset,mp2fit\ndefault, %s/mp2fit'%basis
        return basis_string
    
    
def convert_basis_to_psi4(symb, basis):
    from pyscf.gto.mole import _std_symbol
    '''Convert pyscf internal basis format to Gaussian format string
       Psi4 uses Gaussian 94 format                                 '''
    res = []
    symb = _std_symbol(symb)
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

    # element name
    res.append('%-2s    0' % (symb))

    # gaussian formatting
    for bas in basis:
        for i in range(len(bas[1])-1):
            res.append('%s    %s    1.00' % (SPDF[bas[0]], len(bas[1:])))
            for dat in bas[1:]:
                res.append('%15.9f    %15.9f ' % (dat[0], dat[i+1]))

        #if len(bas[1]) > 2:
        #    for i in range(len(bas[1])-1):
        #        res.append('%s    %s    1.00' % (SPDF[bas[0]], len(bas[1:])))
        #        for dat in bas[1:]:
        #            res.append('%15.9f    %15.9f ' % (dat[0], dat[i+1]))
        #elif len(bas[1]) == 2:
        #    res.append('%s    %s    1.00' % (SPDF[bas[0]], len(bas[1:])))
        #    for dat in bas[1:]:
        #        res.append('%15.9f    %15.9f ' % (dat[0], dat[1]))
        #
        #else:
        #    raise RuntimeError(
        #        'Warning! Please manually check basis set format!')

    # closing
    res.append('****')
    return '\n'.join(res)

_Pyscf2Psi4BasisPermSph = {
   0: [0],
   1: [2, 0, 1],
   #  0  1   2   3   4
   # D0 D+1 D-1 D+2 D-2
   2: [2, 3, 1, 4, 0],
   #  0   1   2  3   4   5   6
   # F0 F+1 F-1 F+2 F-2 F+3 F-3
   3: [3, 4, 2, 5, 1, 6, 0],
   #  0   1   2   3   4   5   6   7   8
   # G0 G+1 G-1 G+2 G-2 G+3 G-3 G+4 G-4
   4: [4, 5, 3, 6, 2, 7, 1, 8, 0],
   #  0   1   2   3   4   5   6   7   8  9   10
   # H0 H+1 H-1 H+2 H-2 H+3 H-3 H+4 H-4 H+5 H-5
   5: [5, 6, 4, 7, 3, 8, 2, 9, 1, 10, 0],
   #  0   1   2   3   4   5   6   7   8   9  10  11  12
   # I0 I+1 I-1 I+2 I-2 I+3 I-3 I+4 I-4 I+5 I-5 I+6 I-6
   6: [6, 7, 5, 8, 4, 9, 3, 10, 2, 11, 1, 12, 0],
}

def ShPyscf2Psi4(mol):
    ###################################################################
    ###   create a index list with the size of AO basis 9/13/17     ###
    ###################################################################
    I_Pyscf2Psi4 = []
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
            I_Pyscf2Psi4 += [(o + iOff) for o in _Pyscf2Psi4BasisPermSph[l]] 
            iOff += 2*l + 1
    I_Pyscf2Psi4 = np.array(I_Pyscf2Psi4)  
    return I_Pyscf2Psi4


psi4_eomcc_template = Template('''#! Psi4 input generated by QSoME
memory $MEMORY MB  # total memory not per thread mempry for parallel jobs

molecule acrolein {
  $CHARGE $SPIN
# acrolein geometry from MD snapshots 
  $GEOM
  units angstrom # default in psi4
  symmetry c1    # no symmetry with embedding potential
  noreorient     # prevent reorienting molecules 
  nocom          # prevent recentering molecules 
}

set {
  roots_per_irrep [3]
  freeze_core false 
  #df_basis_scf aug-cc-pvdz-jkfit
  #df_basis_cc aug-cc-pvdz-ri
  #scf_type df
  #cc_type df
  scf_type pk
  CC_NUM_THREADS $NCORE
}

set cclambda {
  r_convergence 4
}

# EOM-CC3 can only calculate one root at a time
# The highest root is calculated by default
# use PROP_ROOT to assign the desired root
set cceom {
  r_convergence 3
  e_convergence 5
  PROP_ROOT $NROOT 
}

basis {
# generated by PySCF 
$BASIS_TYPE
$BASIS
}

energy('$METHOD')
''')



## Spherical basis function angular momentum ordering
## https://github.com/psi4/psi4/blob/master/psi4/src/psi4/libmints/writer.cc
## https://psicode.org/psi4manual/master/prog_blas.html
#    // Molpro:
#    //  '1s',
#    //  '2px','2py','2pz'
#    //  '3d0','3d2-','3d1+','3d2+','3d1-'
#    //  '4f1+','4f1-','4f0','4f3+','4f2-'
#    //  '4f3-','4f2+'
#    //  '5g0','5g2-','5g1+','5g4+','5g1-','5g2+'
#    //  '5g4-','5g3+','5g3-'
#    //  '6h1+','6h1-','6h2+','6h3+','6h4-','6h3-','6h4+','6h5-','6h0','6h5+','6h2-'
#    //  '7i6+','7i2-','7i5+','7i4+','7i5-','7i2+','7i6-','7i3+','7i4-','7i0','7i3-','7i1-','7i1+'
#
## PySCF ordering
## https://github.com/sunqm/libcint/blob/master/doc/program_ref.pdf
## https://github.com/pyscf/pyscf/issues/1023
## https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics
## follow CCA standard excepts of p orbitals (l=1)
#    // PySCF:
#    // '1s'
#    // '2px','2py','2pz' or '2p1-','2p0','2p1+'
#    // '3d2-','3d1-','3d0','3d1+','3d2+'
#    // '4f3-','4f2-','4f1-','4f0','4f1+','4f2+','4f3+'
#    //  '-l, -l+1, -l+2,..,0,..,l-2,l-1,l'
#
## PSI4 ordering
## https://github.com/MolSSI/QCSchema/issues/45
## https://github.com/psi4/psi4/blob/master/psi4/src/psi4/libmints/writer.cc#L421-L519
#
#
## Gaussian/Molden/PSI4 ordering
## https://gau2grid.readthedocs.io/_/downloads/en/stable/pdf/
#    // Gaussian/Molden/PSI4
#    // '1s'
#    // '2pz','2px','2py' or '2p0','2p1+','2p1-'
#    // '3d0','3d1+','3d1-','3d2+','3d2-'
#    // '4f0','4f1+','4f1-','4f2+','4f2-','4f3+','4f3-'
#    //  '0, 1+, 1-, 2+, 2-, ..., l+, l-'


# Psi4 python script to test the ordering
#import psi4
#import numpy as np
#np.set_printoptions(suppress=True)
#
#
#psi4.set_options({'scf_type': 'pk'})
#
#h2o = psi4.geometry("""
#  0 1
#  H
#  O 1 0.957
#  H 2 0.957 1 104.5
#""")
#
#psi4.basis_helper("""
## CC-pvdz
#spherical
#****
#H     0
#S    3   1.00
#      1.301000D+01           1.968500D-02
#      1.962000D+00           1.379770D-01
#      4.446000D-01           4.781480D-01
#S    1   1.00
#      1.220000D-01           1.000000D+00
#P    1   1.00
#      7.270000D-01           1.0000000
#****
#O     0
#S    8   1.00
#      1.172000D+04           7.100000D-04
#      1.759000D+03           5.470000D-03
#      4.008000D+02           2.783700D-02
#      1.137000D+02           1.048000D-01
#      3.703000D+01           2.830620D-01
#      1.327000D+01           4.487190D-01
#      5.025000D+00           2.709520D-01
#      1.013000D+00           1.545800D-02
#S    8   1.00
#      1.172000D+04          -1.600000D-04
#      1.759000D+03          -1.263000D-03
#      4.008000D+02          -6.267000D-03
#      1.137000D+02          -2.571600D-02
#      3.703000D+01          -7.092400D-02
#      1.327000D+01          -1.654110D-01
#      5.025000D+00          -1.169550D-01
#      1.013000D+00           5.573680D-01
#S    1   1.00
#      3.023000D-01           1.000000D+00
#P    3   1.00
#      1.770000D+01           4.301800D-02
#      3.854000D+00           2.289130D-01
#      1.046000D+00           5.087280D-01
#P    1   1.00
#      2.753000D-01           1.000000D+00
#D    1   1.00
#      1.185000D+00           1.0000000
#****
#""")
#scf_e, wfn = psi4.energy('scf',return_wfn=True)
#F_ao = wfn.Fa_subset("AO").to_array()
#array([[ -0.48136266,  -0.61493728,  -0.36962934,   0.        ,
#          0.5090822 ,  -1.09424183,  -0.44549065,  -0.84186585,
#          0.37987649,   0.        ,  -0.46417694,   0.24712484,
#          0.        ,  -0.28138229,  -0.0223638 ,   0.        ,
#          0.06550476,   0.04399988,   0.        ,  -0.29953392,
#         -0.42998243,  -0.10502384,   0.        ,  -0.39735089],...]


#PySCF scirpt to confirm the ordering
#import pyscf
#from pyscf import gto,scf
#import numpy as np
#np.set_printoptions(suppress=True)
#mol = pyscf.M(
#    atom ='H    0.0   0.756689    -0.520321;O  0.0   0.0     0.065570;H 0    -0.756689    -0.520321',
#    symmetry = False,
#)
#
#
#
#mol.basis = {'H': gto.basis.parse("""
##BASIS SET: (4s,1p) -> [2s,1p]
#H    S
#      1.301000E+01           1.968500E-02           0.000000E+00
#      1.962000E+00           1.379770E-01           0.000000E+00
#      4.446000E-01           4.781480E-01           0.000000E+00
#      1.220000E-01           0.0000000E+00          1.000000E+00
#H    P
#      7.270000E-01           1.0000000
#"""),
#'O': gto.basis.parse("""
##BASIS SET: (9s,4p,1d) -> [3s,2p,1d]
#O    S
#      1.172000E+04           7.100000E-04          -1.600000E-04           0.000000E+00
#      1.759000E+03           5.470000E-03          -1.263000E-03           0.000000E+00
#      4.008000E+02           2.783700E-02          -6.267000E-03           0.000000E+00
#      1.137000E+02           1.048000E-01          -2.571600E-02           0.000000E+00
#      3.703000E+01           2.830620E-01          -7.092400E-02           0.000000E+00
#      1.327000E+01           4.487190E-01          -1.654110E-01           0.000000E+00
#      5.025000E+00           2.709520E-01          -1.169550E-01           0.000000E+00
#      1.013000E+00           1.545800E-02           5.573680E-01           0.000000E+00
#      3.023000E-01           0.0000000E+00          0.0000000E+00          1.000000E+00
#O    P
#      1.770000E+01           4.301800E-02           0.000000E+00
#      3.854000E+00           2.289130E-01           0.000000E+00
#      1.046000E+00           5.087280E-01           0.000000E+00
#      2.753000E-01           0.0000000E+00          1.000000E+00
#O    D
#      1.185000E+00           1.0000000
#""")}
#
#mol.build()
#myhf = scf.RHF(mol)
#myhf.kernel()
#myhf.get_fock()
#array([[ -0.48136265,  -0.61493739,   0.        ,   0.5090822 ,
#         -0.3696292 ,  -1.09424184,  -0.44549045,  -0.84186578,
#          0.        ,  -0.4641769 ,   0.37987652,   0.        ,
#         -0.28138211,   0.24712505,   0.        ,   0.06550478,
#         -0.02236383,  -0.        ,   0.04399993,  -0.29953415,
#         -0.42998266,   0.        ,  -0.3973508 ,  -0.10502371],...]
# use the same contraction from basissetexchange.org
# Psi4 and PySCF fock matrices are identical at the accuracy of 1e-6



if __name__ == '__main__':
    print(dir(Psi4Ext))
