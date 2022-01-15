import random
import copy
import numpy as np
import scipy.sparse as sp
import numba
import torch
from gemnet.training.rotamer_utils import RotamerBase

class DataContainer:
    """
    Parameters
    ----------
        path: str
            Absolute path of the dataset (in npz-format).
        cutoff: float
            Insert edge between atoms if distance is less than cutoff.
        int_cutoff: float
            Cutoff of edge embeddings involved in quadruplet-based message passing.
        triplets_only: bool
            Flag whether to load quadruplet indices as well.
        transforms: list
            Transforms that should be applied on the whole dataset.
        addID: bool
            Whether to add the molecule id to the output.
    """
    def __init__(
        self,
        path,
        cutoff,
        int_cutoff,
        triplets_only=False,
        transforms=None,
        addID=False,
    ):
        self.index_keys = [
            "batch_seg",
            "id_undir",
            "id_swap",
            "id_c",
            "id_a",
            "id3_expand_ba",
            "id3_reduce_ca",
            "Kidx3",
        ]
        if not triplets_only:
            self.index_keys += [
                "id4_int_b",
                "id4_int_a",
                "id4_reduce_ca",
                "id4_expand_db",
                "id4_reduce_cab",
                "id4_expand_abd",
                "Kidx4",
                "id4_reduce_intm_ca",
                "id4_expand_intm_db",
                "id4_reduce_intm_ab",
                "id4_expand_intm_ab",
            ]
        self.triplets_only = triplets_only
        self.cutoff = cutoff
        self.int_cutoff = int_cutoff
        self.addID = addID
        self.keys = ["N", "Z", "R", "F", "E"]
        if addID:
            self.keys += ["id"]

        self._load_npz(path, self.keys)  # set keys as attributes

        if transforms is None:
            self.transforms = []
        else:
            assert isinstance(transforms, (list, tuple))
            self.transforms = transforms

        # modify dataset
        for transform in self.transforms:
            transform(self)

        assert self.R is not None
        assert self.N is not None
        assert self.Z is not None
        assert self.E is not None
        assert self.F is not None

        assert len(self.E) > 0
        assert len(self.F) > 0
        self.process_attributes()


    def process_attributes(self):
        self.E = self.E[:, None]  # shape=(nMolecules,1)
        self.N_cumsum = np.concatenate([[0], np.cumsum(self.N)])

        self.dtypes, dtypes2 = self.get_dtypes()
        self.dtypes.update(dtypes2)  # merge all dtypes in single dict
        self.targets = ["E", "F"]

    def _load_npz(self, path, keys):
        """Load the keys from the file and set as attributes.
        Parameters
        ----------
        path: str
            Absolute path of the dataset (in npz-format).
        keys: list
            Contains keys in the dataset to load and set as attributes.
        Returns
        -------
        None
        """
        with np.load(path, allow_pickle=True) as data:
            for key in keys:
                if key not in data.keys():
                    if key != "F":
                        raise UserWarning(f"Can not find key {key} in the dataset.")
                else:
                    setattr(self, key, data[key])

    @staticmethod
    def _bmat_fast(mats):
        """Combines multiple adjacency matrices into single sparse block matrix.
        Parameters
        ----------
            mats: list
                Has adjacency matrices as elements.
        Returns
        -------
            adj_matrix: sp.csr_matrix
                Combined adjacency matrix (sparse block matrix)
        """
        assert len(mats) > 0
        new_data = np.concatenate([mat.data for mat in mats])

        ind_offset = np.zeros(1 + len(mats), dtype="int32")
        ind_offset[1:] = np.cumsum([mat.shape[0] for mat in mats])
        new_indices = np.concatenate(
            [mats[i].indices + ind_offset[i] for i in range(len(mats))]
        )

        indptr_offset = np.zeros(1 + len(mats))
        indptr_offset[1:] = np.cumsum([mat.nnz for mat in mats])
        new_indptr = np.concatenate(
            [mats[i].indptr[i >= 1 :] + indptr_offset[i] for i in range(len(mats))]
        )

        # Resulting matrix shape: sum of matrices
        shape = (ind_offset[-1], ind_offset[-1])

        # catch case with no edges
        if len(new_data) == 0:
            return sp.csr_matrix(shape)

        return sp.csr_matrix((new_data, new_indices, new_indptr), shape=shape)

    def __len__(self):
        return len(self.N)

    def append_graph_indices(self, data, adj_matrices, adj_matrices_int):
        #### Indices of the moleule structure
        idx_data = {key: None for key in self.index_keys if key != "batch_seg"}
        # Entry A_ij is edge j -> i (!)
        adj_matrix = self._bmat_fast(adj_matrices)
        idx_t, idx_s = adj_matrix.nonzero()  # target and source nodes

        if not self.triplets_only:
            # Entry A_ij is edge j -> i (!)
            adj_matrix_int = self._bmat_fast(adj_matrices_int)
            idx_int_t, idx_int_s = adj_matrix_int.nonzero()  # target and source nodes

        # catch no edge case
        if len(idx_t) == 0:
            for key in idx_data.keys():
                data[key] = np.array([], dtype="int32")
            return self.convert_to_tensor(data)

        # Get mask for undirected edges               0     1      nEdges/2  nEdges/2+1
        # change order of indices such that edge = [[0,1],[0,2], ..., [1,0], [2,0], ...]
        edges = np.stack([idx_t, idx_s], axis=0)
        mask = edges[0] < edges[1]
        edges = edges[:, mask]
        edges = np.concatenate([edges, edges[::-1]], axis=-1).astype("int32")
        idx_t, idx_s = edges[0], edges[1]
        indices = np.arange(len(mask) / 2, dtype="int32")
        idx_data["id_undir"] = np.concatenate(2 * [indices], axis=-1).astype("int32")

        idx_data["id_c"] = idx_s  # node c is source
        idx_data["id_a"] = idx_t  # node a is target

        if not self.triplets_only:
            idx_data["id4_int_a"] = idx_int_t
            idx_data["id4_int_b"] = idx_int_s
        #                                    0         1       ... nEdges/2  nEdges/2+1
        ## swap indices a->c to c->a:   [nEdges/2  nEdges/2+1  ...     0        1 ...   ]
        N_undir_edges = int(len(idx_s) / 2)
        ind = np.arange(N_undir_edges, dtype="int32")
        id_swap = np.concatenate([ind + N_undir_edges, ind])
        idx_data["id_swap"] = id_swap

        # assign an edge_id to each edge
        edge_ids = sp.csr_matrix(
            (np.arange(len(idx_s)), (idx_t, idx_s)),
            shape=adj_matrix.shape,
            dtype="int32",
        )

        #### ------------------------------------ Triplets ------------------------------------ ####
        id3_expand_ba, id3_reduce_ca = self.get_triplets(idx_s, idx_t, edge_ids)
        # embed msg from c -> a with all quadruplets for k and l: c -> a <- k <- l
        # id3_reduce_ca is for k -> a -> c but we want c -> a <- k
        id3_reduce_ca = id_swap[id3_reduce_ca]

        # --------------------- Needed for efficient implementation --------------------- #
        if len(id3_reduce_ca) > 0:
            # id_reduce_ca must be sorted (i.e. grouped would suffice) for ragged_range !
            idx_sorted = np.argsort(id3_reduce_ca)
            id3_reduce_ca = id3_reduce_ca[idx_sorted]
            id3_expand_ba = id3_expand_ba[idx_sorted]
            _, K = np.unique(id3_reduce_ca, return_counts=True)
            idx_data["Kidx3"] = DataContainer.ragged_range(
                K
            )  # K = [1 4 2 3] -> Kidx3 = [0  0 1 2 3  0 1  0 1 2] , (nTriplets,)
        else:
            idx_data["Kidx3"] = np.array([], dtype="int32")
        # ------------------------------------------------------------------------------- #

        idx_data["id3_expand_ba"] = id3_expand_ba  # (nTriplets,)
        idx_data["id3_reduce_ca"] = id3_reduce_ca  # (nTriplets,)

        # node indices in triplet
        # id3_c = idx_s[id3_reduce_ca]
        # id3_a = idx_t[id3_reduce_ca]
        # assert np.all(id3_j == idx_t[id3_expand_ba])
        # id3_b = idx_s[id3_expand_ba]
        #### ---------------------------------------------------------------------------------- ####

        if self.triplets_only:
            data.update(idx_data)
            return self.convert_to_tensor(data)

        #### ----------------------------------- Quadruplets ---------------------------------- ####

        # get quadruplets
        output = self.get_quadruplets(
            idx_s, idx_t, adj_matrix, edge_ids, idx_int_s, idx_int_t
        )
        (
            id4_reduce_ca,
            id4_expand_db,
            id4_reduce_cab,
            id4_expand_abd,
            id4_reduce_intm_ca,
            id4_expand_intm_db,
            id4_reduce_intm_ab,
            id4_expand_intm_ab,
        ) = output

        # --------------------- Needed for efficient implementation --------------------- #
        if len(id4_reduce_ca) > 0:
            # id4_reduce_ca has to be sorted (i.e. grouped would suffice) for ragged range !
            sorted_idx = np.argsort(id4_reduce_ca)
            id4_reduce_ca = id4_reduce_ca[sorted_idx]
            id4_expand_db = id4_expand_db[sorted_idx]
            id4_reduce_cab = id4_reduce_cab[sorted_idx]
            id4_expand_abd = id4_expand_abd[sorted_idx]

            _, K = np.unique(id4_reduce_ca, return_counts=True)
            # K = [1 4 2 3] -> Kidx4 = [0  0 1 2 3  0 1  0 1 2]
            idx_data["Kidx4"] = DataContainer.ragged_range(K)  # (nQuadruplets,)
        else:
            idx_data["Kidx4"] = np.array([], dtype="int32")
        # ------------------------------------------------------------------------------- #

        idx_data["id4_reduce_ca"] = id4_reduce_ca  # (nQuadruplets,)
        idx_data["id4_expand_db"] = id4_expand_db  # (nQuadruplets,)
        idx_data["id4_reduce_cab"] = id4_reduce_cab  # (nQuadruplets,)
        idx_data["id4_expand_abd"] = id4_expand_abd  # (nQuadruplets,)
        idx_data["id4_reduce_intm_ca"] = id4_reduce_intm_ca  # (intmTriplets,)
        idx_data["id4_expand_intm_db"] = id4_expand_intm_db  # (intmTriplets,)
        idx_data["id4_reduce_intm_ab"] = id4_reduce_intm_ab  # (intmTriplets,)
        idx_data["id4_expand_intm_ab"] = id4_expand_intm_ab  # (intmTriplets,)
        idx_data = {k:v for k,v in idx_data.items() if v is not None}
        data.update(idx_data)
        return data

    def __getitem__(self, idx):
        """
        GYF: it is pretty slow to 
        calculate these idxs and angles on the fly! 
        It is nearly impossible to feed into the model.
        For example, a data has 1964 atoms
        would have id4_int_a in shape [288896], 
        id4_expand_abd in shape [169469540]
        Parameters
        ----------
            idx: array-like
                Ids of the molecules to get.
        Returns
        -------
            data: dict
                nMolecules = len(idx)
                nAtoms = total sum of atoms in the selected molecules
                Contains following keys and values:
                - id: np.ndarray, shape (nMolecules,)
                    Ids of the molecules in the dataset.
                - N: np.ndarray, shape (nMolecules,)
                    Number of atoms in the molecules.
                - Z: np.ndarray, shape (nAtoms,)
                    Atomic numbers (dt. Ordnungszahl).
                - R: np.ndarray, shape (nAtoms,3)
                    Atom positions in °A.
                - F: np.ndarray, shape (nAtoms,3)
                    Forces at the atoms in eV/°A.
                - E: np.ndarray, shape (nMolecules,1)
                    Energy of the molecule in eV.
                - batch_seg: np.ndarray, shape (nAtoms,)
                    Contains the index of the sample the atom belongs to.
                    E.g. [0,0,0, 1,1,1,1, 2,...] where first molecule has 3 atoms,
                    second molecule has 4 atoms etc.
                - id_c: np.ndarray, shape (nEdges,)
                    Indices of edges' source atom.
                - id_a: np.ndarray, shape (nEdges,)
                    Indices of edges' target atom.
                - id_undir: np.ndarray, shape (nEdges,)
                    Indices where the same index denotes opposite edges, c-> and a->c.
                - id_swap: np.ndarray, shape (nEdges,)
                    Indices to map c->a to a->c.
                - id3_expand_ba: np.ndarray, shape (nTriplets,)
                    Indices to map the edges from c->a to b->a in the triplet-based massage passing.
                - id3_reduce_ca: np.ndarray, shape (nTriplets,)
                    Indices to map the edges from c->a to c->a in the triplet-based massage passing.
                - Kidx3: np.ndarray, shape (nTriplets,)
                    Indices to reshape the neighbor indices b->a into a dense matrix.
                - id4_int_a: np.ndarray, shape (nInterEdges,)
                    Indices of the atom a of the interaction edge.
                - id4_int_b: np.ndarray, shape (nInterEdges,)
                    Indices of the atom b of the interaction edge.
                - id4_reduce_ca: np.ndarray, shape (nQuadruplets,)
                    Indices to map c->a to c->a in quadruplet-message passing.
                - id4_expand_db: np.ndarray, shape (nQuadruplets,)
                    Indices to map c->a to d->b in quadruplet-message passing.
                - id4_reduce_intm_ca: np.ndarray, shape (intmTriplets,)
                    Indices to map c->a to intermediate c->a.
                - id4_expand_intm_db: np.ndarray, shape (intmTriplets,)
                    Indices to map d->b to intermediate d->b.
                - id4_reduce_intm_ab: np.ndarray, shape (intmTriplets,)
                    Indices to map b-a to intermediate b-a of the quadruplet's part c->a-b.
                - id4_expand_intm_ab: np.ndarray, shape (intmTriplets,)
                    Indices to map b-a to intermediate b-a of the quadruplet's part a-b<-d.
                - id4_reduce_cab: np.ndarray, shape (nQuadruplets,)
                    Indices to map from intermediate c->a to quadruplet c->a.
                - id4_expand_abd: np.ndarray, shape (nQuadruplets,)
                    Indices to map from intermediate d->b to quadruplet d->b.
                - Kidx4: np.ndarray, shape (nTriplets,)
                    Indices to reshape the neighbor indices d->b into a dense matrix.
        """
        if isinstance(idx, (int, np.int64, np.int32)):
            idx = [idx]
        if isinstance(idx, tuple):
            idx = list(idx)
        if isinstance(idx, slice):
            idx = np.arange(idx.start, min(idx.stop, len(self)), idx.step)

        data = {}
        if self.addID:
            data["id"] = self.id[idx]
        data["E"] = self.E[idx]
        data["N"] = self.N[idx]
        data["batch_seg"] = np.repeat(np.arange(len(idx), dtype=np.int32), data["N"])

        data["Z"] = np.zeros(np.sum(data["N"]), dtype=np.int32)
        data["R"] = np.zeros([np.sum(data["N"]), 3], dtype=np.float32)
        data["F"] = np.zeros([np.sum(data["N"]), 3], dtype=np.float32)

        nend = 0
        adj_matrices = []
        adj_matrices_int = []
        for k, i in enumerate(idx):
            n = data["N"][k]
            nstart = nend
            nend = nstart + n
            s, e = (
                self.N_cumsum[i],
                self.N_cumsum[i + 1],
            )  # start and end idx of atoms belonging to molecule

            data["F"][nstart:nend] = self.F[s:e]
            data["Z"][nstart:nend] = self.Z[s:e]
            R = self.R[s:e]
            data["R"][nstart:nend] = R

            D_ij = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)
            # get adjacency matrix for embeddings
            adj_mat = sp.csr_matrix(D_ij <= self.cutoff)
            adj_mat -= sp.eye(n, dtype=np.bool)
            adj_matrices.append(adj_mat)

            if not self.triplets_only:
                # get adjacency matrix for interaction
                adj_mat = sp.csr_matrix(D_ij <= self.int_cutoff)
                adj_mat -= sp.eye(n, dtype=np.bool)
                adj_matrices_int.append(adj_mat)

        data = self.append_graph_indices(data, adj_matrices, adj_matrices_int)
        return self.convert_to_tensor(data)

    @staticmethod
    def get_triplets(idx_s, idx_t, edge_ids):
        """
        Get triplets c -> a <- b
        """
        # Edge indices of triplets k -> a -> i
        id3_expand_ba = edge_ids[idx_s].data.astype("int32").flatten()
        id3_reduce_ca = edge_ids[idx_s].tocoo().row.astype("int32").flatten()

        id3_i = idx_t[id3_reduce_ca]
        id3_k = idx_s[id3_expand_ba]
        mask = id3_i != id3_k
        id3_expand_ba = id3_expand_ba[mask]
        id3_reduce_ca = id3_reduce_ca[mask]

        return id3_expand_ba, id3_reduce_ca

    @staticmethod
    def get_quadruplets(idx_s, idx_t, adj_matrix, edge_ids, idx_int_s, idx_int_t):
        """
        c -> a - b <- d where D_ab <= int_cutoff; D_ca & D_db <= cutoff
        """
        # Number of incoming edges to target and source node of interaction edges
        nNeighbors_t = adj_matrix[idx_int_t].sum(axis=1).A1.astype("int32")
        nNeighbors_s = adj_matrix[idx_int_s].sum(axis=1).A1.astype("int32")
        id4_reduce_intm_ca = (
            edge_ids[idx_int_t].data.astype("int32").flatten()
        )  # (intmTriplets,)
        id4_expand_intm_db = (
            edge_ids[idx_int_s].data.astype("int32").flatten()
        )  # (intmTriplets,)
        # note that id4_reduce_intm_ca and id4_expand_intm_db have the same shape but
        # id4_reduce_intm_ca[i] and id4_expand_intm_db[i] may not belong to the same interacting quadruplet !

        # each reduce edge (c->a) has to be repeated as often as there are neighbors for node b
        # vice verca for the edges of the source node (d->b) and node a
        id4_reduce_cab = DataContainer.repeat_blocks(
            nNeighbors_t, nNeighbors_s
        )  # (nQuadruplets,)
        id4_reduce_ca = id4_reduce_intm_ca[id4_reduce_cab]  # intmTriplets -> nQuadruplets

        N = np.repeat(nNeighbors_t, nNeighbors_s)
        id4_expand_abd = np.repeat(
            np.arange(len(id4_expand_intm_db)), N
        )  # (nQuadruplets,)
        id4_expand_db = id4_expand_intm_db[id4_expand_abd]  # intmTriplets -> nQuadruplets

        id4_reduce_intm_ab = np.repeat(
            np.arange(len(idx_int_t)), nNeighbors_t
        )  # (intmTriplets,)
        id4_expand_intm_ab = np.repeat(
            np.arange(len(idx_int_t)), nNeighbors_s
        )  # (intmTriplets,)

        # Mask out all quadruplets where nodes appear more than once
        idx_c = idx_s[id4_reduce_ca]
        idx_a = idx_t[id4_reduce_ca]
        idx_b = idx_t[id4_expand_db]
        idx_d = idx_s[id4_expand_db]

        mask1 = idx_c != idx_b
        mask2 = idx_a != idx_d
        mask3 = idx_c != idx_d
        mask = mask1 * mask2 * mask3  # logical and

        id4_reduce_ca = id4_reduce_ca[mask]
        id4_expand_db = id4_expand_db[mask]
        id4_reduce_cab = id4_reduce_cab[mask]
        id4_expand_abd = id4_expand_abd[mask]

        return (
            id4_reduce_ca,
            id4_expand_db,
            id4_reduce_cab,
            id4_expand_abd,
            id4_reduce_intm_ca,
            id4_expand_intm_db,
            id4_reduce_intm_ab,
            id4_expand_intm_ab,
        )

    def convert_to_tensor(self, data):
        for key in data:
            data[key] = torch.tensor(data[key], dtype=self.dtypes[key])
        return data

    def get_dtypes(self):
        """
        Returns
        -------
            dtypes: tuple
                (dtypes_input, dtypes_target) TF input types for the inputs and targets
                stored in dicts.
        """
        # dtypes of dataset values
        dtypes_input = {}
        if self.addID:
            dtypes_input["id"] = torch.int64
        dtypes_input["Z"] = torch.int64
        dtypes_input["N"] = torch.int64
        dtypes_input["R"] = torch.float32
        for key in self.index_keys:
            dtypes_input[key] = torch.int64

        dtypes_target = {}
        dtypes_target["E"] = torch.float32
        dtypes_target["F"] = torch.float32

        return dtypes_input, dtypes_target

    @staticmethod
    @numba.njit(nogil=True)
    def repeat_blocks(sizes, repeats):
        """Repeat blocks of indices.
        From https://stackoverflow.com/questions/51154989/numpy-vectorized-function-to-repeat-blocks-of-consecutive-elements
        Examples
        --------
            sizes = [1,3,2] ; repeats = [3,2,3]
            Return: [0 0 0  1 2 3 1 2 3  4 5 4 5 4 5]
            sizes = [0,3,2] ; repeats = [3,2,3]
            Return: [0 1 2 0 1 2  3 4 3 4 3 4]
            sizes = [2,3,2] ; repeats = [2,0,2]
            Return: [0 1 0 1  5 6 5 6]
        """
        a = np.arange(np.sum(sizes))
        indices = np.empty((sizes * repeats).sum(), dtype=np.int32)
        start = 0
        oi = 0
        for i, size in enumerate(sizes):
            end = start + size
            for _ in range(repeats[i]):
                oe = oi + size
                indices[oi:oe] = a[start:end]
                oi = oe
            start = end
        return indices

    @staticmethod
    @numba.njit(nogil=True)
    def ragged_range(sizes):
        """
        -------
        Example
        -------
            sizes = [1,3,2] ;
            Return: [0  0 1 2  0 1]
        """
        a = np.arange(sizes.max())
        indices = np.empty(sizes.sum(), dtype=np.int32)
        start = 0
        for size in sizes:
            end = start + size
            indices[start:end] = a[:size]
            start = end
        return indices

    def collate_fn(self, batch):
        """
        custom batching function because batches have variable shape
        """
        batch = batch[0]  # already batched
        inputs = {}
        targets = {}
        for key in batch:
            if key in self.targets:
                targets[key] = batch[key]
            else:
                inputs[key] = batch[key]
        return inputs, targets

class PairDataContainer(object):
    """
    This class is a wrapper class of Data Container
    """
    def __init__(self, container1, container2):
        self.container1 = container1
        self.container2 = container2
        assert len(container1) == len(container2)

    def collate_fn(self, batch):
        batch = batch[0]  # already batched
        inputs = list()
        targets = list()
        for element in batch:
            input = dict()
            target = dict()
            for key in element:
                if key in self.targets:
                    target[key] = element[key]
                else:
                    input[key] = element[key]
            inputs.append(input)
            targets.append(target)
        return inputs, targets

    def __len__(self):
        return len(self.container1)

    def __getattr__(self, name):   
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.container1, name)

    def __getitem__(self, idx):
        return [self.container1[idx], self.container2[idx]]


class EBMDataContainer(DataContainer):
    """
    Data Container for EBM Training
    """
    def __init__(
        self,
        path,
        cutoff,
        int_cutoff,
        triplets_only=False,
        transforms=None,
        addID=False,
        num_neighbors=64,
        num_negative=1,
        rotamer_library_path="./data/dunbrack-rotamer/original"
    ):
        self.index_keys = [
            "batch_seg",
            "id_undir",
            "id_swap",
            "id_c",
            "id_a",
            "id3_expand_ba",
            "id3_reduce_ca",
            "Kidx3",
            "sampled_residue",
            "residue_ids", 
            "residue_types", 
            "atom_ids"
        ]
        if not triplets_only:
            self.index_keys += [
                "id4_int_b",
                "id4_int_a",
                "id4_reduce_ca",
                "id4_expand_db",
                "id4_reduce_cab",
                "id4_expand_abd",
                "Kidx4",
                "id4_reduce_intm_ca",
                "id4_expand_intm_db",
                "id4_reduce_intm_ab",
                "id4_expand_intm_ab",
            ]
        self.triplets_only = triplets_only
        self.cutoff = cutoff
        self.int_cutoff = int_cutoff
        self.addID = addID
        self.num_neighbors = num_neighbors
        self.num_negative = num_negative
        self.rotamer = RotamerBase(library_path=rotamer_library_path)
        self.keys = ["N", "Z", "R", "F", "E", "residue_ids", "residue_types", "atom_ids"]
        if addID:
            self.keys += ["id"]

        self._load_npz(path, self.keys)  # set keys as attributes

        if transforms is None:
            self.transforms = []
        else:
            assert isinstance(transforms, (list, tuple))
            self.transforms = transforms

        # modify dataset
        for transform in self.transforms:
            transform(self)

        assert self.R is not None
        assert self.N is not None
        assert self.Z is not None
        assert self.E is not None
        assert self.F is not None
        assert self.residue_ids is not None  # (N_atoms,)
        assert self.residue_types is not None  # (N_residues,)
        assert self.atom_ids is not None   # (N_atoms,)

        assert len(self.E) > 0
        assert len(self.F) > 0
        self.process_attributes()

    def sample_residue(self, idx):
        """
        Return sampled residues. 
        If no valid residue, the default value is -1.
        """
        sampled_residues = np.full(len(idx), -1, dtype=np.int32)
        for k, protein_id in enumerate(idx):
            start = self.N_cumsum[protein_id]
            end = self.N_cumsum[protein_id + 1]
            residue_ids = self.residue_ids[start:end]
            # We sample from all the residues of a protein except the first and last ones
            residue_start, residue_end = residue_ids[0] + 1, residue_ids[-1]
            residue_index_range = np.arange(residue_start, residue_end) # [N_residue-2, ]
            residue_types = self.residue_types[residue_start:residue_end] # [N_residue-2, ]
            in_library = residue_types < len(self.rotamer.amino_acids)
            assert in_library.sum() > 0 # if none of the residues are in lib, then raise error
            residue_index_range = residue_index_range[in_library]
            sampled_residue_index = np.random.choice(residue_index_range, 1)[0]
            # return the index of the residue rather than the type of the residue
            sampled_residues[k] = sampled_residue_index
        return sampled_residues

    def get_topk_atoms(self, protein_id, residue_id, num_atom, new_atom_positions=None):
        """
        Paramters:
            protein_id: (int)
            residue_id: (int) the index of the sampled residue in the data array (not the type of the residue)
            num_atom: (int) selected num atom 
        """
        start = self.N_cumsum[protein_id]
        end = self.N_cumsum[protein_id + 1]
        atom_positions = new_atom_positions if new_atom_positions is not None else self.R[start:end]
        residue_ids = self.residue_ids[start:end]
        atom_ids = self.atom_ids[start:end]
        target_residue_CB = (residue_ids == residue_id) & (atom_ids == 4)
        target_residue_CB_position = atom_positions[target_residue_CB]
        assert target_residue_CB_position.shape[0] == 1
        all_dist = ((atom_positions - target_residue_CB_position) ** 2).sum(-1)
        is_target_residue = (residue_ids == residue_id)
        all_dist[is_target_residue] = 0.0
        topk_indices = np.argsort(all_dist)[:num_atom] + start
        return topk_indices

    def compute_Phi(self, atom_positions, residue_ids, atom_ids, target_residue_id):
        """
        Paramters:
            atom_positions: [N_atoms, 3]
            residue_ids: [N_atoms,]
            atom_ids: [N_atoms,]
            target_residue_id: (int) 
        """
        C_before_position = atom_positions[(residue_ids == target_residue_id - 1) & (atom_ids == 2)]
        N_target_position = atom_positions[(residue_ids == target_residue_id) & (atom_ids == 0)]
        CA_target_position = atom_positions[(residue_ids == target_residue_id) & (atom_ids == 1)]
        C_target_position = atom_positions[(residue_ids == target_residue_id) & (atom_ids == 2)]
        assert C_before_position.shape[0] == 1 and N_target_position.shape[0] == 1 \
               and CA_target_position.shape[0] == 1 and C_target_position.shape[0] == 1
        Phi = self.rotamer.compute_dihedral(C_before_position, N_target_position, CA_target_position,
                                            C_target_position)[0]
        return Phi

    def compute_Psi(self, atom_positions, residue_ids, atom_ids, target_residue_id):
        N_target_position = atom_positions[(residue_ids == target_residue_id) & (atom_ids == 0)]
        CA_target_position = atom_positions[(residue_ids == target_residue_id) & (atom_ids == 1)]
        C_target_position = atom_positions[(residue_ids == target_residue_id) & (atom_ids == 2)]
        N_next_position = atom_positions[(residue_ids == target_residue_id + 1) & (atom_ids == 0)]
        assert N_target_position.shape[0] == 1 and CA_target_position.shape[0] == 1 \
               and C_target_position.shape[0] == 1 and N_next_position.shape[0] == 1
        Psi = self.rotamer.compute_dihedral(N_target_position, CA_target_position, C_target_position,
                                            N_next_position)[0]
        return Psi

    def compute_Chis(self, atom_positions, residue_ids, atom_ids, target_residue_id):
        target_residue_name = self.rotamer.amino_acids[self.residue_types[target_residue_id]]
        Chis = np.zeros(4, dtype=np.float32)
        for Chi_id, Chi_atoms in enumerate(self.rotamer.chis_atoms[target_residue_name]):
            atom_1, atom_2, atom_3, atom_4 = Chi_atoms
            atom_1_position = atom_positions[(residue_ids == target_residue_id) & (atom_ids == atom_1)]
            atom_2_position = atom_positions[(residue_ids == target_residue_id) & (atom_ids == atom_2)]
            atom_3_position = atom_positions[(residue_ids == target_residue_id) & (atom_ids == atom_3)]
            atom_4_position = atom_positions[(residue_ids == target_residue_id) & (atom_ids == atom_4)]
            assert atom_1_position.shape[0] == 1 and atom_2_position.shape[0] == 1 and \
                   atom_3_position.shape[0] == 1 and atom_4_position.shape[0] == 1
            Chi = self.rotamer.compute_dihedral(atom_1_position, atom_2_position, atom_3_position, atom_4_position)[0]
            Chis[Chi_id] = Chi
        return Chis

    def rotate_single_side_chain(self, protein_id, residue_id):
        start = self.N_cumsum[protein_id]
        end = self.N_cumsum[protein_id + 1]
        atom_positions = self.R[start:end]
        residue_ids = self.residue_ids[start:end]
        atom_ids = self.atom_ids[start:end]

        # Compute Phi, Psi and Chis; Sample new Chis based on the rotamer library
        Phi = self.compute_Phi(atom_positions, residue_ids, atom_ids, residue_id)  # scalar
        Psi = self.compute_Psi(atom_positions, residue_ids, atom_ids, residue_id)  # scalar
        Chis = self.compute_Chis(atom_positions, residue_ids, atom_ids, residue_id)  # (4,)
        target_Chis = self.rotamer.sample_chis(Phi, Psi,
                                                self.rotamer.amino_acids[self.residue_types[residue_id]])  # (4,)

        # Rotate side chain according to the sampled Chis
        atom_positions_ = atom_positions[residue_ids == residue_id]
        atom_ids_ = atom_ids[residue_ids == residue_id]
        residue_name = self.rotamer.amino_acids[self.residue_types[residue_id]]
        all_chi_atoms = self.rotamer.chis_atoms[residue_name]
        rotated_atom_positions_ = self.rotamer.rotate_side_chain(Chis, target_Chis,
                                                                 atom_positions_, atom_ids_, all_chi_atoms)
        rotated_atom_positions = copy.deepcopy(atom_positions)
        rotated_atom_positions[residue_ids == residue_id] = rotated_atom_positions_
        return rotated_atom_positions

    def get_positive_data(self, idx):
        data = {}
        if self.addID:
            data["id"] = self.id[idx]
        data["E"] = self.E[idx]
        data["N"] = np.min(np.stack([self.N[idx], np.full(len(idx), self.num_neighbors, dtype=np.int32)],
                                    axis=1), axis=1)
        data["batch_seg"] = np.repeat(np.arange(len(idx), dtype=np.int32), data["N"])
        data["Z"] = np.zeros(np.sum(data["N"]), dtype=np.int32)
        data["R"] = np.zeros([np.sum(data["N"]), 3], dtype=np.float32)
        data["F"] = np.zeros([np.sum(data["N"]), 3], dtype=np.float32)
        data["sampled_residue"] = self.sample_residue(idx)

        nend = 0
        adj_matrices = []
        adj_matrices_int = []
        for protein_id, residue_id, num_atom in zip(idx, data["sampled_residue"], data["N"]):
            topk_indices = self.get_topk_atoms(protein_id, residue_id, num_atom)
            nstart = nend
            nend = nstart + num_atom
            data["F"][nstart:nend] = self.F[topk_indices]
            data["Z"][nstart:nend] = self.Z[topk_indices]
            R = self.R[topk_indices]
            data["R"][nstart:nend] = R

            D_ij = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)
            # get adjacency matrix for embeddings
            adj_mat = sp.csr_matrix(D_ij <= self.cutoff)
            adj_mat -= sp.eye(num_atom, dtype=np.bool)
            adj_matrices.append(adj_mat)

            if not self.triplets_only:
                # get adjacency matrix for interaction
                adj_mat = sp.csr_matrix(D_ij <= self.int_cutoff)
                adj_mat -= sp.eye(num_atom, dtype=np.bool)
                adj_matrices_int.append(adj_mat)

        return data, adj_matrices, adj_matrices_int

    def get_negative_data(self, positive_data, idx):
        all_negative_data = []
        adj_matrices = []
        adj_matrices_int = []
        for negative_id in range(self.num_negative):
            negative_data = {}
            if self.addID:
                negative_data["id"] = positive_data["id"]
            negative_data["E"] = positive_data["E"]
            negative_data["N"] = positive_data["N"]
            negative_data["batch_seg"] = positive_data["batch_seg"] + len(idx) * (negative_id + 1)
            negative_data["sampled_residue"] = positive_data["sampled_residue"]
            negative_data["Z"] = np.zeros(np.sum(negative_data["N"]), dtype=np.int32)
            negative_data["R"] = np.zeros([np.sum(negative_data["N"]), 3], dtype=np.float32)
            negative_data["F"] = np.zeros([np.sum(negative_data["N"]), 3], dtype=np.float32)

            nend = 0
            for protein_id, sampled_residue_index, num_atom in zip(idx, negative_data["sampled_residue"], negative_data["N"]):
                rotated_atom_positions = self.rotate_single_side_chain(protein_id, sampled_residue_index)
                topk_indices = self.get_topk_atoms(protein_id, sampled_residue_index, num_atom,
                                                   new_atom_positions=rotated_atom_positions)
                nstart = nend
                nend = nstart + num_atom
                negative_data["F"][nstart:nend] = self.F[topk_indices]
                negative_data["Z"][nstart:nend] = self.Z[topk_indices]
                R = rotated_atom_positions[topk_indices - self.N_cumsum[protein_id]]
                negative_data["R"][nstart:nend] = R

                D_ij = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)
                # get adjacency matrix for embeddings
                adj_mat = sp.csr_matrix(D_ij <= self.cutoff)
                adj_mat -= sp.eye(num_atom, dtype=np.bool)
                adj_matrices.append(adj_mat)

                if not self.triplets_only:
                    # get adjacency matrix for interaction
                    adj_mat = sp.csr_matrix(D_ij <= self.int_cutoff)
                    adj_mat -= sp.eye(num_atom, dtype=np.bool)
                    adj_matrices_int.append(adj_mat)
            all_negative_data.append(negative_data)

        return all_negative_data, adj_matrices, adj_matrices_int

    def __getitem__(self, idx):
        try:
            if isinstance(idx, (int, np.int64, np.int32)):
                idx = [idx]
            if isinstance(idx, tuple):
                idx = list(idx)
            if isinstance(idx, slice):
                idx = np.arange(idx.start, min(idx.stop, len(self)), idx.step)

            # Get the positive conformation
            all_data = []
            all_adj_matrices = []
            all_adj_matrices_int = []
            positive_data, positive_adj_matrices, positive_adj_matrices_int = self.get_positive_data(idx)
            all_data.append(positive_data)
            all_adj_matrices += (positive_adj_matrices)
            all_adj_matrices_int += (positive_adj_matrices_int)

            # Get the negative conformations
            negative_data, negative_adj_matrices, negative_adj_matrices_int = self.get_negative_data(positive_data, idx)
            all_data += negative_data
            all_adj_matrices += negative_adj_matrices
            all_adj_matrices_int += negative_adj_matrices_int

            # Pack positive and negative data
            data = {}
            for key in positive_data.keys():
                value = np.concatenate([data_[key] for data_ in all_data], axis=0)
                data[key] = value

            # Graph construction
            data = self.append_graph_indices(data, all_adj_matrices, all_adj_matrices_int)
            data = self.convert_to_tensor(data)
            return data
        except Exception as e:
            if isinstance(idx, (int, np.int64, np.int32)):
                idx = [idx]
            if isinstance(idx, tuple):
                idx = list(idx)
            if isinstance(idx, slice):
                idx = np.arange(idx.start, min(idx.stop, len(self)), idx.step)
            return self.__getitem__([(x + 1) % self.__len__() for x in idx])

    @classmethod
    def from_config(cls, config):
        return cls(config.dataset, cutoff=config.model.cutoff, int_cutoff=config.model.int_cutoff, 
                     triplets_only=config.model.triplets_only, addID=True, 
                     num_neighbors=config.num_neighbors, num_negative=config.num_negative, 
                    rotamer_library_path=config.rotamer_library_path)


class PairDataContainer(object):
    """
    This class is a wrapper class of Data Container

    """
    def __init__(self, container1, container2):
        self.container1 = container1
        self.container2 = container2
        assert len(container1) == len(container2)

    def collate_fn(self, batch):
        batch = batch[0]  # already batched
        inputs = list()
        targets = list()
        for element in batch:
            input = dict()
            target = dict()
            for key in element:
                if key in self.targets:
                    target[key] = element[key]
                else:
                    input[key] = element[key]
            inputs.append(input)
            targets.append(target)
        return inputs, targets

    def __len__(self):
        return len(self.container1)

    def __getattr__(self, name):   
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.container1, name)

    def __getitem__(self, idx):
        return [self.container1[idx], self.container2[idx]]

    @classmethod
    def from_config(cls, config):
        wt_data_container = DataContainer(
        config.dataset_wt, cutoff=config.model.cutoff, int_cutoff=config.model.int_cutoff, triplets_only=config.model.triplets_only, 
        addID=True)

        mt_data_container = DataContainer(
        config.dataset_mt, cutoff=config.model.cutoff, int_cutoff=config.model.int_cutoff, triplets_only=config.model.triplets_only, 
        addID=True)
        return cls(wt_data_container, mt_data_container)