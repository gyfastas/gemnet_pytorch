import numpy as np
import torch

from .base_layers import Dense


class AtomEmbedding(torch.nn.Module):
    """
    Initial atom embeddings based on the atom type

    Parameters
    ----------
        emb_size: int
            Atom embeddings size
        chain_embedding_scheme: "sum" or "mean"
        residue_embedding_scheme: "add", "concat" or "expand_vocab"
    """

    eps = 1e-6
    num_chain_type = 2
    raw_atom_size = 93
    local_atom_size = 40
    residue_size = 30

    def __init__(self, emb_size, name=None, start_from=1, add_residue_embedding=False, add_chain_embedding=False,
                 chain_embedding_scheme="sum", residue_embedding_scheme="add"):
        super().__init__()
        emb_size = emb_size // 2 if add_residue_embedding and residue_embedding_scheme == "concat" else emb_size
        emb_size = emb_size // 2 if add_chain_embedding else emb_size
        self.emb_size = emb_size
        self.start_from = start_from
        self.add_residue_embedding = add_residue_embedding
        self.add_chain_embedding = add_chain_embedding
        self.chain_embedding_scheme = chain_embedding_scheme
        self.residue_embedding_scheme = residue_embedding_scheme
        # Atom embeddings: We go up to Pu (94). Use 93 dimensions because of 0-based indexing
        self.atom_embeddings = torch.nn.Embedding(self.raw_atom_size, emb_size)
        if add_residue_embedding and residue_embedding_scheme == "expand_vocab":
            self.residue_embeddings = torch.nn.Embedding(self.local_atom_size * self.residue_size, emb_size)
        else:
            self.residue_embeddings = torch.nn.Embedding(self.residue_size, emb_size)
        # init by uniform distribution
        torch.nn.init.uniform_(self.atom_embeddings.weight, a=-np.sqrt(3), b=np.sqrt(3))
        torch.nn.init.uniform_(self.residue_embeddings.weight, a=-np.sqrt(3), b=np.sqrt(3))

    def forward(self, Z, residue_types, chain_ids):
        """
        Returns
        -------
            h: Tensor, shape=(nAtoms, emb_size)
                Atom embeddings.
        """
        h = self.atom_embeddings(Z - self.start_from)  # -1 because Z.min()=1 (==Hydrogen)
        if self.add_residue_embedding:
            if self.residue_embedding_scheme == "concat":
                residue_embedding = self.residue_embeddings(residue_types)
                h = torch.cat([h, residue_embedding], dim=-1)
            elif self.residue_embedding_scheme == "add":
                residue_embedding = self.residue_embeddings(residue_types)
                h += residue_embedding
            elif self.residue_embedding_scheme == "expand_vocab":
                local_atom_types = residue_types * self.local_atom_size + Z - self.start_from
                h = self.residue_embeddings(local_atom_types)
            else:
                raise ValueError("Residue embedding scheme {} is not supported.".format(self.residue_embedding_scheme))
        if self.add_chain_embedding:
            sum_chain_embedding = torch.zeros((self.num_chain_type, h.shape[-1]), dtype=torch.float32, device=h.device).scatter_add_(
                0, chain_ids.unsqueeze(-1).repeat(1, h.shape[-1]), h)
            chain_cnt = torch.zeros(self.num_chain_type, dtype=torch.float32, device=h.device).scatter_add_(
                0, chain_ids, torch.ones(chain_ids.shape[0], dtype=torch.float32, device=h.device))
            mean_chain_embedding = sum_chain_embedding / (chain_cnt.unsqueeze(-1) + self.eps)
            if self.chain_embedding_scheme == "sum":
                chain_embedding = sum_chain_embedding[chain_ids]
            elif self.chain_embedding_scheme == "mean":
                chain_embedding = mean_chain_embedding[chain_ids]
            else:
                raise ValueError("Chain embedding scheme {} is not supported.".format(self.chain_embedding_scheme))
            h = torch.cat([h, chain_embedding], dim=-1)

        return h


class EdgeEmbedding(torch.nn.Module):
    """
    Edge embedding based on the concatenation of atom embeddings and subsequent dense layer.

    Parameters
    ----------
        atom_features: int
            Embedding size of the atom embeddings.
        edge_features: int
            Embedding size of the edge embeddings.
        out_features: int
            Embedding size after the dense layer.
        activation: str
            Activation function used in the dense layer.
    """

    def __init__(
        self, atom_features, edge_features, out_features, activation=None, name=None
    ):
        super().__init__()
        in_features = 2 * atom_features + edge_features
        self.dense = Dense(in_features, out_features, activation=activation, bias=False)

    def forward(self, h, m_rbf, idnb_a, idnb_c,):
        """
        Returns
        -------
            m_ca: Tensor, shape=(nEdges, emb_size)
                Edge embeddings.
        """
        # m_rbf: shape (nEdges, nFeatures)
        # in embedding block: m_rbf = rbf ; In interaction block: m_rbf = m_ca

        h_a = h[idnb_a]  # shape=(nEdges, emb_size)
        h_c = h[idnb_c]  # shape=(nEdges, emb_size)

        m_ca = torch.cat([h_a, h_c, m_rbf], dim=-1)  # (nEdges, 2*emb_size+nFeatures)
        m_ca = self.dense(m_ca)  # (nEdges, emb_size)
        return m_ca
