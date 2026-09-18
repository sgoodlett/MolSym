import numpy as np
from .function_set import FunctionSet

class CartesianCoordinates(FunctionSet):
    """
    FunctionSet for Cartesian coordinates
    """
    def __init__(self, symtext) -> None:
        # xyz on each atom in molecule
        # fxn map is ncart (natom sets of xyz) by nsymel
        fxn_list = [i for i in range(3*len(symtext.mol))]
        super().__init__(symtext, fxn_list)

    def get_fxn_map(self):
        """
        Builds the function map for all of the Cartesian coordinates under each symmetry element.
        
        :rtype: NumPy array of shape (nsymels, 3, 3)
        """
        # Symel by xyz by xyz, maps xyz to xyz under symels
        fxn_map = np.zeros((len(self.symtext), 3, 3))
        #phase_map = None
        for s in range(len(self.symtext)):
            fxn_map[s, :, :] = self.symtext.symels[s].rrep.T
        return fxn_map#, phase_map
 
    def get_symmetry_equiv_functions(self):
        """
        Finds the sets of functions that are invariant under all of the symmetry elements.

        :rtype: List[List[int]]
        """
        symm_equiv = []
        done = []
        xyz = np.array([0,1,2], dtype=int)
        for atom_i in range(len(self.symtext.mol)):
            for xyz_idx in range(3):
                fidx = 3*atom_i + xyz_idx
                if fidx in done:
                    continue
                equiv_set = []
                for sidx in range(len(self.symtext)):
                    newatom = self.symtext.atom_map[atom_i, sidx]
                    notzero = xyz[~np.isclose(self.symtext.symels[sidx].rrep[:,xyz_idx], 0.0, atol=1e-10)]
                    r = [3*newatom+i for i in notzero]
                    equiv_set += r
                reduced_equiv_set = list(set(equiv_set))
                symm_equiv.append(reduced_equiv_set)
                done += reduced_equiv_set
        return symm_equiv

    def special_function(self, salc, coord, sidx, irrmat):
        """
        Defines how to map an internal coordinate under a symmetry operation for the ProjectionOp function.
        
        :type salc: NumPy array of shape (number of internal coordinates,)
        :type coord: int
        :type sidx: int
        :type irrmat: NumPy array of shape (nsymel, irrep.d, irrep.d)
        """
        atom_idx = self.symtext.atom_map[coord//3, sidx]
        cfxn = coord % 3
        xyz = self.fxn_map[sidx,cfxn,:]
        for i in range(3):
            if self.symtext.complex:
                salc[:,:,3*atom_idx+i] += np.conj(irrmat[sidx, :, :]) * xyz[i]
            else:
                salc[:,:,3*atom_idx+i] += irrmat[sidx, :, :] * xyz[i]
        return salc

class LinearCartesian(FunctionSet):
    def __init__(self, symtext):
        fxn_list = [i for i in range(3*len(symtext.mol))]
        super().__init__(symtext, fxn_list)

    def get_fxn_map(self):
        return 0

    def get_symmetry_equiv_functions(self):
        symm_equiv = []
        if self.symtext.pg.family =="C":
            for atom_i in range(len(self.symtext.mol)):
                symm_equiv.append([3*atom_i, 3*atom_i+1]) # x,y
                symm_equiv.append([3*atom_i+2]) # z
        elif self.symtext.pg.family =="D":
            done = []
            for atom_i in range(len(self.symtext.mol)):
                if atom_i not in done:
                    atom_j = self.symtext.atom_map[atom_i,1]
                    if atom_i == atom_j:
                        done.append(atom_i)
                        symm_equiv.append([3*atom_i, 3*atom_i+1]) # x,y
                        symm_equiv.append([3*atom_i+2]) # z
                    else:
                        done.append(atom_i)
                        done.append(atom_j)
                        symm_equiv.append([3*atom_i, 3*atom_i+1, 3*atom_j, 3*atom_j+1]) # x,y
                        symm_equiv.append([3*atom_i+2, 3*atom_j+2]) # z
        return symm_equiv

    def special_function(self, salc, coord, sidx, irrmat):
        if self.symtext.pg.family == "C":
            if irrmat.symbol not in ["Sigma^+","Pi"]:
                return salc
            if sidx == 1:
                det = -1 # Reflections have a negative determinant
            for se_fxns in self.SE_fxns:
                if coord in se_fxns:
                    coord_se_fxns = se_fxns
                    break
            if len(se_fxns) == 1 and irrmat.symbol != "Sigma^+":
                return salc
            elif len(se_fxns) == 2 and irrmat.symbol != "Pi":
                return salc
            for idx, f in enumerate(coord_se_fxns):
                salc[idx,idx,f] += 1.0
            return salc
        elif self.symtext.pg.family == "D":
            if sidx > 1:
                return salc # Treat sidx as E and i (0,1) and ignore all others
            if irrmat.symbol not in ["Sigma_g^+", "Sigma_u^+","Pi_g", "Pi_u"]:
                return salc
            coord_atom_idx = coord // 3
            for se_fxns in self.SE_fxns:
                if coord in se_fxns:
                    coord_se_fxns = se_fxns
                    break
            central_atom = self.symtext.atom_map[coord_atom_idx,1] == coord_atom_idx
            if central_atom:
                if coord % 3 == 2 and irrmat.symbol == "Sigma_u^+":
                    salc[0,0,coord] += 1
                elif coord % 3 != 2 and irrmat.symbol == "Pi_u":
                    salc[0,0,3*coord_atom_idx] += 1
                    salc[1,1,3*coord_atom_idx+1] += 1
            else:
                other_atom = self.symtext.atom_map[coord_atom_idx, 1]
                other_coord = (3*other_atom) + (coord%3)
                if coord % 3 == 2 and irrmat.symbol == "Sigma_g^+":
                    salc[0,0,coord] += 1
                    salc[0,0,other_coord] += -1
                elif coord % 3 == 2 and irrmat.symbol == "Sigma_u^+":
                    salc[0,0,coord] += 1
                    salc[0,0,other_coord] += 1
                elif coord % 3 != 2 and irrmat.symbol == "Pi_g":
                    salc[0,0,3*coord_atom_idx] += 1
                    salc[0,0,3*other_atom] += -1
                    salc[1,1,3*coord_atom_idx+1] += 1
                    salc[1,1,3*other_atom+1] += -1
                elif coord % 3 != 2 and irrmat.symbol == "Pi_u":
                    salc[0,0,3*coord_atom_idx] += 1
                    salc[0,0,3*other_atom] +=  1
                    salc[1,1,3*coord_atom_idx+1] += 1
                    salc[1,1,3*other_atom+1] +=  1
            return salc
        else:
            raise ValueError("Bad point group.")

