import torch
from torch.utils.checkpoint import checkpoint


def get_config_model(Ns):
    return {
        "em": {'N0': 37, 'N1': Ns},
        "sum": [
            {'Ns': Ns, 'Nh': 2, 'Nk': 3, 'nn': 8},
            {'Ns': Ns, 'Nh': 2, 'Nk': 3, 'nn': 8},
            {'Ns': Ns, 'Nh': 2, 'Nk': 3, 'nn': 8},
            {'Ns': Ns, 'Nh': 2, 'Nk': 3, 'nn': 8},
            {'Ns': Ns, 'Nh': 2, 'Nk': 3, 'nn': 8},
            {'Ns': Ns, 'Nh': 2, 'Nk': 3, 'nn': 8},
            {'Ns': Ns, 'Nh': 2, 'Nk': 3, 'nn': 8},
            {'Ns': Ns, 'Nh': 2, 'Nk': 3, 'nn': 8},
            {'Ns': Ns, 'Nh': 2, 'Nk': 3, 'nn': 16},
            {'Ns': Ns, 'Nh': 2, 'Nk': 3, 'nn': 16},
            {'Ns': Ns, 'Nh': 2, 'Nk': 3, 'nn': 16},
            {'Ns': Ns, 'Nh': 2, 'Nk': 3, 'nn': 16},
            {'Ns': Ns, 'Nh': 2, 'Nk': 3, 'nn': 16},
            {'Ns': Ns, 'Nh': 2, 'Nk': 3, 'nn': 16},
            {'Ns': Ns, 'Nh': 2, 'Nk': 3, 'nn': 16},
            {'Ns': Ns, 'Nh': 2, 'Nk': 3, 'nn': 16},
            {'Ns': Ns, 'Nh': 2, 'Nk': 3, 'nn': 32},
            {'Ns': Ns, 'Nh': 2, 'Nk': 3, 'nn': 32},
            {'Ns': Ns, 'Nh': 2, 'Nk': 3, 'nn': 32},
            {'Ns': Ns, 'Nh': 2, 'Nk': 3, 'nn': 32},
            {'Ns': Ns, 'Nh': 2, 'Nk': 3, 'nn': 32},
            {'Ns': Ns, 'Nh': 2, 'Nk': 3, 'nn': 32},
            {'Ns': Ns, 'Nh': 2, 'Nk': 3, 'nn': 32},
            {'Ns': Ns, 'Nh': 2, 'Nk': 3, 'nn': 32},
            {'Ns': Ns, 'Nh': 2, 'Nk': 3, 'nn': 64},
            {'Ns': Ns, 'Nh': 2, 'Nk': 3, 'nn': 64},
            {'Ns': Ns, 'Nh': 2, 'Nk': 3, 'nn': 64},
            {'Ns': Ns, 'Nh': 2, 'Nk': 3, 'nn': 64},
            {'Ns': Ns, 'Nh': 2, 'Nk': 3, 'nn': 64},
            {'Ns': Ns, 'Nh': 2, 'Nk': 3, 'nn': 64},
            {'Ns': Ns, 'Nh': 2, 'Nk': 3, 'nn': 64},
            {'Ns': Ns, 'Nh': 2, 'Nk': 3, 'nn': 64},
        ],
        "spl": {'N0': Ns, 'N1': 32, 'Nh': 4},
        "dm": {'N0': 32, 'N1': 32, 'N2': 7}
    }


# >> UTILS
def unpack_state_features(X, ids_topk, q):
    # compute displacement vectors
    R_nn = X[ids_topk - 1] - X.unsqueeze(1)
    # compute distance matrix
    D_nn = torch.norm(R_nn, dim=2)
    # mask distances
    D_nn = D_nn + torch.max(D_nn) * (D_nn < 1e-2).float()
    # normalize displacement vectors
    R_nn = R_nn / D_nn.unsqueeze(2)

    # prepare sink
    q = torch.cat([torch.zeros((1, q.shape[1]), device=q.device), q], dim=0)
    ids_topk = torch.cat([torch.zeros((1, ids_topk.shape[1]), dtype=torch.long, device=ids_topk.device), ids_topk],
                         dim=0)
    D_nn = torch.cat([torch.zeros((1, D_nn.shape[1]), device=D_nn.device), D_nn], dim=0)
    R_nn = torch.cat([torch.zeros((1, R_nn.shape[1], R_nn.shape[2]), device=R_nn.device), R_nn], dim=0)

    return q, ids_topk, D_nn, R_nn


# >>> OPERATIONS
class StateUpdate(torch.nn.Module):
    def __init__(self, Ns, Nh, Nk):
        super(StateUpdate, self).__init__()
        # operation parameters
        self.Ns = Ns
        self.Nh = Nh
        self.Nk = Nk

        # node query model
        self.nqm = torch.nn.Sequential(
            torch.nn.Linear(2 * Ns, Ns),
            torch.nn.ELU(),
            torch.nn.Linear(Ns, Ns),
            torch.nn.ELU(),
            torch.nn.Linear(Ns, 2 * Nk * Nh),
        )

        # edges scalar keys model
        self.eqkm = torch.nn.Sequential(
            torch.nn.Linear(6 * Ns + 1, Ns),
            torch.nn.ELU(),
            torch.nn.Linear(Ns, Ns),
            torch.nn.ELU(),
            torch.nn.Linear(Ns, Nk),
        )

        # edges vector keys model
        self.epkm = torch.nn.Sequential(
            torch.nn.Linear(6 * Ns + 1, Ns),
            torch.nn.ELU(),
            torch.nn.Linear(Ns, Ns),
            torch.nn.ELU(),
            torch.nn.Linear(Ns, 3 * Nk),
        )

        # edges value model
        self.evm = torch.nn.Sequential(
            torch.nn.Linear(6 * Ns + 1, 2 * Ns),
            torch.nn.ELU(),
            torch.nn.Linear(2 * Ns, 2 * Ns),
            torch.nn.ELU(),
            torch.nn.Linear(2 * Ns, 2 * Ns),
        )

        # scalar projection model
        self.qpm = torch.nn.Sequential(
            torch.nn.Linear(Nh * Ns, Ns),
            torch.nn.ELU(),
            torch.nn.Linear(Ns, Ns),
            torch.nn.ELU(),
            torch.nn.Linear(Ns, Ns),
        )

        # vector projection model
        self.ppm = torch.nn.Sequential(
            torch.nn.Linear(Nh * Ns, Ns, bias=False),
        )

        # scaling factor for attention
        self.sdk = torch.nn.Parameter(torch.sqrt(torch.tensor(Nk).float()), requires_grad=False)

    def forward(self, q, p, q_nn, p_nn, d_nn, r_nn):
        # q: [N, S]
        # p: [N, 3, S]
        # q_nn: [N, n, S]
        # p_nn: [N, n, 3, S]
        # d_nn: [N, n]
        # r_nn: [N, n, 3]
        # N: number of nodes
        # n: number of nearest neighbors
        # S: state dimensions
        # H: number of attention heads

        # get dimensions
        N, n, S = q_nn.shape

        # node inputs packing
        X_n = torch.cat([
            q,
            torch.norm(p, dim=1),
        ], dim=1)  # [N, 2*S]

        # edge inputs packing
        X_e = torch.cat([
            d_nn.unsqueeze(2),  # distance
            X_n.unsqueeze(1).repeat(1, n, 1),  # centered state
            q_nn,  # neighbors states
            torch.norm(p_nn, dim=2),  # neighbors vector states norms
            torch.sum(p.unsqueeze(1) * r_nn.unsqueeze(3), dim=2),  # centered vector state projections
            torch.sum(p_nn * r_nn.unsqueeze(3), dim=2),  # neighbors vector states projections
        ], dim=2)  # [N, n, 6*S+1]

        # node queries
        Q = self.nqm.forward(X_n).view(N, 2, self.Nh, self.Nk)  # [N, 2*S] -> [N, 2, Nh, Nk]

        # scalar edges keys while keeping interaction order inveriance
        Kq = self.eqkm.forward(X_e).view(N, n, self.Nk).transpose(1, 2)  # [N, n, 6*S+1] -> [N, Nk, n]

        # vector edges keys while keeping bond order inveriance
        Kp = torch.cat(torch.split(self.epkm.forward(X_e), self.Nk, dim=2), dim=1).transpose(1, 2)

        # edges values while keeping interaction order inveriance
        V = self.evm.forward(X_e).view(N, n, 2, S).transpose(1, 2)  # [N, n, 6*S+1] -> [N, 2, n, S]

        # vectorial inputs packing
        Vp = torch.cat([
            V[:, 1].unsqueeze(2) * r_nn.unsqueeze(3),
            p.unsqueeze(1).repeat(1, n, 1, 1),
            p_nn,
            # torch.cross(p.unsqueeze(1).repeat(1,n,1,1), r_nn.unsqueeze(3).repeat(1,1,1,S), dim=2),
        ], dim=1).transpose(1, 2)  # [N, 3, 3*n, S]

        # queries and keys collapse
        Mq = torch.nn.functional.softmax(torch.matmul(Q[:, 0], Kq) / self.sdk, dim=2)  # [N, Nh, n]
        Mp = torch.nn.functional.softmax(torch.matmul(Q[:, 1], Kp) / self.sdk, dim=2)  # [N, Nh, 3*n]

        # scalar state attention mask and values collapse
        Zq = torch.matmul(Mq, V[:, 0]).view(N, self.Nh * self.Ns)  # [N, Nh*S]
        Zp = torch.matmul(Mp.unsqueeze(1), Vp).view(N, 3, self.Nh * self.Ns)  # [N, 3, Nh*S]

        # decode outputs
        qh = self.qpm.forward(Zq)
        ph = self.ppm.forward(Zp)

        # update state with residual
        qz = q + qh
        pz = p + ph

        return qz, pz


def state_max_pool(q, p, M):
    # get norm of state vector
    s = torch.norm(p, dim=2)  # [N, S]

    # perform mask pool on mask
    q_max, _ = torch.max(M.unsqueeze(2) * q.unsqueeze(1), dim=0)  # [n, S]
    _, s_ids = torch.max(M.unsqueeze(2) * s.unsqueeze(1), dim=0)  # [n, S]

    # get maximum state vector
    p_max = torch.gather(p, 0, s_ids.unsqueeze(2).repeat((1, 1, p.shape[2])))

    return q_max, p_max


class StatePoolLayer(torch.nn.Module):
    def __init__(self, N0, N1, Nh):
        super(StatePoolLayer, self).__init__()
        # state attention model
        self.sam = torch.nn.Sequential(
            torch.nn.Linear(2 * N0, N0),
            torch.nn.ELU(),
            torch.nn.Linear(N0, N0),
            torch.nn.ELU(),
            torch.nn.Linear(N0, 2 * Nh),
        )

        # attention heads decoding
        self.zdm = torch.nn.Sequential(
            torch.nn.Linear(Nh * N0, N0),
            torch.nn.ELU(),
            torch.nn.Linear(N0, N0),
            torch.nn.ELU(),
            torch.nn.Linear(N0, N1),
        )

        # vector attention heads decoding
        self.zdm_vec = torch.nn.Sequential(
            torch.nn.Linear(Nh * N0, N1, bias=False)
        )

    def forward(self, q, p, M):
        # create filter for softmax
        F = (1.0 - M + 1e-6) / (M - 1e-6)

        # pack features
        z = torch.cat([q, torch.norm(p, dim=1)], dim=1)

        # multiple attention pool on state
        Ms = torch.nn.functional.softmax(self.sam.forward(z).unsqueeze(1) + F.unsqueeze(2), dim=0).view(M.shape[0],
                                                                                                        M.shape[1], -1,
                                                                                                        2)
        qh = torch.matmul(torch.transpose(q, 0, 1), torch.transpose(Ms[:, :, :, 0], 0, 1))
        ph = torch.matmul(torch.transpose(torch.transpose(p, 0, 2), 0, 1),
                          torch.transpose(Ms[:, :, :, 1], 0, 1).unsqueeze(1))

        # attention heads decoding
        qr = self.zdm.forward(qh.view(Ms.shape[1], -1))
        pr = self.zdm_vec.forward(ph.view(Ms.shape[1], p.shape[1], -1))

        return qr, pr


# >>> LAYERS
class StateUpdateLayer(torch.nn.Module):
    def __init__(self, layer_params):
        super(StateUpdateLayer, self).__init__()
        # define operation
        self.su = StateUpdate(*[layer_params[k] for k in ['Ns', 'Nh', 'Nk']])
        # store number of nearest neighbors
        self.m_nn = torch.nn.Parameter(torch.arange(layer_params['nn'], dtype=torch.int64), requires_grad=False)

    def forward(self, Z):
        # unpack input
        q, p, ids_topk, D_topk, R_topk = Z

        # update q, p
        ids_nn = ids_topk[:, self.m_nn]
        # q, p = self.su.forward(q, p, q[ids_nn], p[ids_nn], D_topk[:,self.m_nn], R_topk[:,self.m_nn])

        # with checkpoint
        q = q.requires_grad_()
        p = p.requires_grad_()
        q, p = checkpoint(self.su.forward, q, p, q[ids_nn], p[ids_nn], D_topk[:, self.m_nn], R_topk[:, self.m_nn])

        # sink
        q[0] = q[0] * 0.0
        p[0] = p[0] * 0.0

        return q, p, ids_topk, D_topk, R_topk


class CrossStateUpdateLayer(torch.nn.Module):
    def __init__(self, layer_params):
        super(CrossStateUpdateLayer, self).__init__()
        # get cross-states update layer parameters
        Ns = layer_params['Ns']
        self.Nh = layer_params['cNh']
        self.Nk = layer_params['cNk']

        # atomic level state update layers
        self.sul = StateUpdateLayer(layer_params)

        # queries model
        self.cqm = torch.nn.Sequential(
            torch.nn.Linear(2 * Ns, Ns),
            torch.nn.ELU(),
            torch.nn.Linear(Ns, Ns),
            torch.nn.ELU(),
            torch.nn.Linear(Ns, self.Nk * self.Nh),
        )

        # keys model
        self.ckm = torch.nn.Sequential(
            torch.nn.Linear(2 * Ns, Ns),
            torch.nn.ELU(),
            torch.nn.Linear(Ns, Ns),
            torch.nn.ELU(),
            torch.nn.Linear(Ns, self.Nk),
        )

        # values model
        self.cvm = torch.nn.Sequential(
            torch.nn.Linear(2 * Ns, Ns),
            torch.nn.ELU(),
            torch.nn.Linear(Ns, Ns),
            torch.nn.ELU(),
            torch.nn.Linear(Ns, Ns),
        )

        # projection heads
        self.cpm = torch.nn.Sequential(
            torch.nn.Linear((self.Nh + 1) * Ns, Ns),
            torch.nn.ELU(),
            torch.nn.Linear(Ns, Ns),
            torch.nn.ELU(),
            torch.nn.Linear(Ns, Ns),
        )

        # scaling factor for attention
        self.sdk = torch.nn.Parameter(torch.sqrt(torch.tensor(self.Nk).float()), requires_grad=False)

    def forward(self, Z):
        # unpack input
        q0, p0, ids0_topk, D0_nn, R0_nn = Z[0]
        q1, p1, ids1_topk, D1_nn, R1_nn = Z[1]

        # forward independently
        qa0, pz0, _, _, _ = self.sul.forward((q0, p0, ids0_topk, D0_nn, R0_nn))
        qa1, pz1, _, _, _ = self.sul.forward((q1, p1, ids1_topk, D1_nn, R1_nn))

        # pack states
        s0 = torch.cat([qa0, torch.norm(pz0, dim=1)], dim=1)
        s1 = torch.cat([qa1, torch.norm(pz1, dim=1)], dim=1)

        # compute queries
        Q0 = self.cqm.forward(s0).reshape(s0.shape[0], self.Nh, self.Nk)
        Q1 = self.cqm.forward(s1).reshape(s1.shape[0], self.Nh, self.Nk)

        # compute keys
        K0 = self.ckm.forward(s0).transpose(0, 1)
        K1 = self.ckm.forward(s1).transpose(0, 1)

        # compute values
        V0 = self.cvm.forward(s0)
        V1 = self.cvm.forward(s1)

        # transform 1 -> 0
        M10 = torch.nn.functional.softmax(torch.matmul(Q0, K1 / self.sdk), dim=2)
        qh0 = torch.matmul(M10, V1).view(Q0.shape[0], -1)

        # transform 0 -> 1
        M01 = torch.nn.functional.softmax(torch.matmul(Q1, K0 / self.sdk), dim=2)
        qh1 = torch.matmul(M01, V0).view(Q1.shape[0], -1)

        # projections and residual
        # qz0 = qa0 + self.cpm.forward(qh0)
        # qz1 = qa1 + self.cpm.forward(qh1)
        qz0 = self.cpm.forward(torch.cat([qa0, qh0], dim=1))
        qz1 = self.cpm.forward(torch.cat([qa1, qh1], dim=1))

        return ((qz0, pz0, ids0_topk, D0_nn, R0_nn), (qz1, pz1, ids1_topk, D1_nn, R1_nn))


def get_top_k_m_batch(batch, k=64):
    """
    Extract top_k from pos and M : (N,N_r) from consecutive indices
    :param batch:
    :param k:
    :return:
    """
    with torch.no_grad():
        graphs = batch.to_data_list()
        offset = 0
        offset_residue = 0
        all_top_k = []
        all_m = []
        atoms_sizes = []
        residue_sizes = []
        # Automatic batching does not work for variable created post graph
        for graph in graphs:
            dists = torch.cdist(graph.pos, graph.pos)
            top_k = dists.topk(k=k, largest=False).indices + offset
            all_top_k.append(top_k)
            atoms_sizes.append(top_k)
            offset += len(dists)

            all_m.append(graph.atom_to_res_map + offset_residue)
            residue_size = int(graph.atom_to_res_map.max())
            offset_residue += residue_size
            residue_sizes.append(residue_size)
        all_top_k = torch.vstack(all_top_k)
        all_m = torch.hstack(all_m).long() - 1
        M = torch.zeros((len(all_m), int(all_m.max() + 1)))
        M[torch.arange(len(all_m)), all_m] = 1
    return all_top_k, M, atoms_sizes, residue_sizes


class PestoModel(torch.nn.Module):
    def __init__(self, config):
        super(PestoModel, self).__init__()
        # features encoding models for structures and library
        self.em = torch.nn.Sequential(
            torch.nn.Linear(config['em']['N0'], config['em']['N1']),
            torch.nn.ELU(),
            torch.nn.Linear(config['em']['N1'], config['em']['N1']),
            torch.nn.ELU(),
            torch.nn.Linear(config['em']['N1'], config['em']['N1']),
        )
        # atomic level state update model
        self.sum = torch.nn.Sequential(*[StateUpdateLayer(layer_params) for layer_params in config['sum']])

        # atomic to residue reduction layer
        self.spl = StatePoolLayer(config['spl']['N0'], config['spl']['N1'], config['spl']['Nh'])

        # decoding mlp
        # self.dm = torch.nn.Sequential(
        #     torch.nn.Linear(2 * config['dm']['N0'], config['dm']['N1']),
        #     torch.nn.ELU(),
        #     torch.nn.Linear(config['dm']['N1'], config['dm']['N1']),
        #     torch.nn.ELU(),
        #     torch.nn.Linear(config['dm']['N1'], config['dm']['N2']),
        # )

    def forward(self, graph, surface):
        """

        :param X: position of the nodes
        :param ids_topk: neighbors id
        :param q0: initial node features
        :param M: atom to residue map : binary tensor of shape (N, Nr)
        :return:
        """
        X, q0 = graph.pos, graph.x
        ids_topk, M, atom_sizes, residue_sizes = get_top_k_m_batch(graph)

        # encode features
        q = self.em.forward(q0)

        # initial state vectors
        p0 = torch.zeros((q.shape[0] + 1, X.shape[1], q.shape[1]), device=X.device)

        # unpack state features with sink
        q, ids_topk, D_nn, R_nn = unpack_state_features(X, ids_topk, q)

        # atomic tsa layers
        qa, pa, _, _, _ = self.sum.forward((q, p0, ids_topk, D_nn, R_nn))

        # atomic to residue attention pool (without sink)
        qr, pr = self.spl.forward(qa[1:], pa[1:], M)

        # decode state
        zr = torch.cat([qr, torch.norm(pr, dim=1)], dim=1)

        # z = self.dm.forward(zr)
        split_embs = torch.split(zr, residue_sizes, dim=0)

        return split_embs
