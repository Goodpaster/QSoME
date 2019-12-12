# Need a method in multiple places? Write one here!
# Daniel Graham

from pyscf import gto
import functools
import time

def concat_mols(subsys_list):
    """Concatenates Mole objects into one Mole.

    Parameters
    ----------
    subsys_list : list
        List of subsystems to concatenate into single Mole.
    """

    assert (len(subsys_list) > 1),"Must have more than 1 subsystem"

    mol1 = gto.mole.copy(subsys_list[0].mol)
    for j in range(mol1.natm):
        old_name = mol1.atom_symbol(j)
        new_name = mol1.atom_symbol(j) + '-0'
        mol1._atom[j] = (new_name, mol1._atom[j][1])
        if old_name in mol1._basis.keys():
            mol1._basis[new_name] = mol1._basis.pop(old_name)
        if old_name in mol1.ecp.keys():
            mol1.ecp[new_name] = mol1.ecp.pop(old_name)
            mol1._ecp[new_name] = mol1._ecp.pop(old_name)
            mol1._atm, mol1._ecpbas, mol1._env = mol1.make_ecp_env(mol1._atm, mol1._ecp, mol1._env)

    #if subsys_list[0].flip_ros:
    #    mol1.spin *= -1
    #    mol1.build()
    for n in range(1, len(subsys_list)):
        mol2 = gto.mole.copy(subsys_list[n].mol)
        #if subsys_list[n].flip_ros:
        #    mol2.spin *= -1
        #    mol2.build()
        for j in range(mol2.natm):
            old_name = mol2.atom_symbol(j)
            new_name = mol2.atom_symbol(j) + '-' + str(n)
            mol2._atom[j] = (new_name, mol2._atom[j][1])
            if old_name in mol2._basis.keys():
                mol2._basis[new_name] = mol2._basis.pop(old_name)
            if old_name in mol2.ecp.keys():
                mol2.ecp[new_name] = mol2.ecp.pop(old_name)
                mol2._ecp[new_name] = mol2._ecp.pop(old_name)
                mol2._atm, mol2._ecpbas, mol2._env = mol2.make_ecp_env(mol2._atm, mol2._ecp, mol2._env)
        mol1 = gto.mole.conc_mol(mol1, mol2)

    #Remove overlapping ghost atoms.
    # I still think there is a better way.
    def remove_overlap(atom_list):
        added_already = {}
        no_dup = []
        for i in range(len(atom_list)):
            coord_tuple = tuple(atom_list[i][1])
            atom_name = atom_list[i][0]
            if not 'ghost' in atom_name:
                no_dup.append(atom_list[i])
                if coord_tuple in added_already:
                    dup_index = added_already[coord_tuple]
                    if not 'ghost' in no_dup[dup_index][0]:
                        print ("OVERLAPPING ATOMS")
                    else:
                        del no_dup[added_already[coord_tuple]]
                        for key in added_already.keys():
                            if added_already[key] > added_already[coord_tuple]:
                                added_already[key] -= 1

                added_already[coord_tuple] = (len(no_dup) - 1)
            else:
                if coord_tuple in added_already:
                    pass 
                else:
                    no_dup.append(atom_list[i])
                    added_already[coord_tuple] = (len(no_dup) - 1)
        return no_dup

    mol1._atom = remove_overlap(mol1._atom)    
    mol1.atom = mol1._atom
    mol1.unit = 'bohr'
    mol1.build(basis=mol1._basis)
    return mol1

def time_method(function_name=None):
    def real_decorator(func):
        @functools.wraps(func)
        def wrapper_time_method(*args, **kwargs):
            ts = time.time()
            result = func(*args, **kwargs)
            te = time.time()
            if function_name is None:
                name = func.__name__.upper()
            else:
                name = function_name
            elapsed_t = (te - ts)
            print( f'TIMING: {name}'.ljust(40) + f'{elapsed_t:>39.4f}s')
            return result
        return wrapper_time_method 
    return real_decorator
