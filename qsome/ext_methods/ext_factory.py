from qsome.ext_methods.molpro_ext import MolproExt
from qsome.ext_methods.openmolcas_ext import OpenMolCASExt
#from ext_methods.bagel_ext import BagelExt

class ExtFactory:
    def get_ext_obj(self, ext_prgm, mol_obj, method_name, ext_pot, core_ham=None, filename='temp', work_dir=None, scr_dir=None, nproc=None, pmem=None, save_orbs=False, save_density=False, hl_dict=None):
        if ext_prgm == 'molpro':
            return MolproExt(mol_obj, method_name, ext_pot, core_ham, filename, work_dir, scr_dir, nproc, pmem, save_orbs, save_density, hl_dict)
        elif ext_prgm == 'molcas' or ext_prgm == 'openmolcas':
            return OpenMolCASExt(mol_obj, method_name, ext_pot)
        #elif ext_prgm == 'bagel':
        #    return BagelExt(mol_obj, method_name, ext_pot)
