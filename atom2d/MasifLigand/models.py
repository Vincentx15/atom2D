import torch
import torch.nn as nn
from torch_scatter import scatter

from base_nets import DiffusionNetBatch, GraphDiffNetParallel, GraphDiffNetSequential, GraphDiffNetAttention, \
    GraphDiffNetBipartite, AtomNetGraph, PestoModel, get_config_model


class MasifLigandNet(torch.nn.Module):
    def __init__(self, in_channels=5, in_channels_surf=5, out_channel=64, C_width=128, N_block=4, with_gradient_features=True,
                 linear_sizes=(128,), dropout=0.3, use_mean=False, batch_norm=False, use_graph=False,
                 use_graph_only=False, use_pesto=False, pesto_width=16,
                 output_graph=False, graph_model='parallel', use_gat=False, use_v2=False,
                 use_skip=False, neigh_th=8, flash=True, use_mp=False, out_features=1, use_distance=False,
                 use_wln=False, **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.in_channels_surf = in_channels_surf
        self.out_channel = out_channel
        self.use_mean = use_mean

        # Create the model
        self.use_graph = use_graph or use_graph_only
        self.use_graph_only = use_graph_only
        self.use_pesto = use_pesto
        if use_graph_only:
            if self.use_pesto:
                cfg = get_config_model(pesto_width)
                self.encoder_model = PestoModel(cfg)
                self.top_net_graph = nn.Sequential(*[
                    nn.ReLU(),
                    nn.Linear(64, pesto_width * 2),
                    nn.ReLU(),
                    nn.Dropout(p=0.25),
                    nn.Linear(pesto_width * 2, out_features)
                ])
            else:
                self.encoder_model = AtomNetGraph(C_in=in_channels,
                                                  C_out=out_channel,
                                                  C_width=C_width,
                                                  last_factor=4,
                                                  use_distance=use_distance)
                self.top_net_graph = nn.Sequential(*[
                    nn.ReLU(),
                    nn.Linear(C_width * 4, C_width * 2),
                    nn.ReLU(),
                    nn.Dropout(p=0.25),
                    nn.Linear(C_width * 2, out_features)
                ])
        elif not use_graph:
            self.encoder_model = DiffusionNetBatch(C_in=in_channels_surf,
                                                   C_out=out_channel,
                                                   C_width=C_width,
                                                   N_block=N_block,
                                                   last_activation=None,
                                                   with_gradient_features=with_gradient_features,
                                                   use_bn=batch_norm,
                                                   dropout=False)
        else:
            if graph_model == 'parallel':
                self.encoder_model = GraphDiffNetParallel(C_in=in_channels,
                                                          C_out=out_channel,
                                                          C_width=C_width,
                                                          N_block=N_block,
                                                          last_activation=torch.relu,
                                                          with_gradient_features=with_gradient_features,
                                                          use_mp=use_mp,
                                                          use_bn=batch_norm,
                                                          output_graph=output_graph,
                                                          dropout=dropout,
                                                          use_distance=use_distance)
            elif graph_model == 'sequential':
                self.encoder_model = GraphDiffNetSequential(C_in=in_channels,
                                                            C_out=out_channel,
                                                            C_width=C_width,
                                                            N_block=N_block,
                                                            last_activation=torch.relu,
                                                            with_gradient_features=with_gradient_features,
                                                            use_mp=use_mp,
                                                            use_gat=use_gat,
                                                            use_skip=use_skip,
                                                            use_bn=batch_norm,
                                                            output_graph=output_graph,
                                                            dropout=dropout,
                                                            use_distance=use_distance)
            elif graph_model == 'attention':
                self.encoder_model = GraphDiffNetAttention(C_in=in_channels,
                                                           C_out=out_channel,
                                                           C_width=C_width,
                                                           N_block=N_block,
                                                           last_activation=torch.relu,
                                                           with_gradient_features=with_gradient_features,
                                                           use_bn=batch_norm,
                                                           flash=flash,
                                                           dropout=dropout,
                                                           use_distance=use_distance)
            elif graph_model == 'bipartite':
                self.encoder_model = GraphDiffNetBipartite(C_in_graph=in_channels,
                                                           C_in_surf=in_channels_surf,
                                                           C_out=out_channel,
                                                           C_width=C_width,
                                                           N_block=N_block,
                                                           with_gradient_features=with_gradient_features,
                                                           last_activation=torch.relu,
                                                           use_bn=batch_norm,
                                                           output_graph=output_graph,
                                                           use_gat=use_gat,
                                                           use_v2=use_v2,
                                                           use_skip=use_skip,
                                                           use_wln=use_wln,
                                                           neigh_th=neigh_th,
                                                           dropout=dropout,
                                                           use_distance=use_distance)
        # Top FCs
        in_features = out_channel
        self.top_net = nn.Sequential(*[
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(in_features // 2, out_features=out_features)
        ])
        # todo: check this
        self.top_net = nn.Sequential(*[
            nn.Linear(in_features, in_features),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(in_features),
            nn.SiLU(),
            nn.Linear(in_features, out_features=out_features)
        ])

        self.feat_extractor = FeatExtractor()

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, batch):
        """
        Both inputs should unwrap as (features, confidence, vertices, mass, L, evals, evecs, gradX, gradY, faces)
        pairs_loc are the coordinates of points shape (n_pairs, 2, 3)
        :param x_left:
        :param x_right:
        :return:
        """

        graph, surface = None, None

        if not self.use_graph_only:
            surface = batch.surface
        if self.use_graph:
            graph = batch.graph

        # HMR features
        features = []
        for cf, gf, nd in zip(surface.chem_feats, surface.geom_feats, surface.nbr_vids):
            features.append(self.feat_extractor(cf, gf, nd))
        surface.x = features

        # forward pass
        processed = self.encoder_model(graph=graph, surface=surface)
        # todo: check this
        x = torch.stack([torch.mean(x, dim=-2) for x in processed])
        x = self.top_net(x)
        return x

        # if self.use_graph_only:
        #     if self.use_pesto:
        #         graph_embs = []
        #         for individual_emb in processed:
        #             x = torch.max(individual_emb, dim=-2).values
        #             x = self.top_net_graph(x).view(-1)
        #             graph_embs.append(x)
        #         return torch.stack(graph_embs)
        #     else:
        #         graph_embs = []
        #         graph.x = processed
        #         for individual_graph in graph.to_data_list():
        #             x = torch.max(individual_graph.x, dim=-2).values
        #             x = self.top_net_graph(x).view(-1)
        #             graph_embs.append(x)
        #         return torch.cat(graph_embs)
        # else:
        #     if self.use_mean:
        #         x = torch.stack([torch.mean(x, dim=-2) for x in processed])
        #     else:
        #         x = torch.stack([torch.max(x, dim=-2).values for x in processed])
        #     x = self.top_net(x)
        # return x


class FeatExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        h_dim = 128
        dropout = 0.1
        num_gdf = 16
        num_signatures = 16

        # chem feat
        chem_feat_dim = 2 + num_gdf * 2
        self.chem_mlp = nn.Sequential(
            nn.Linear(chem_feat_dim, h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, 2 * h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(2 * h_dim),
        )
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

        # geom feats
        geom_input_dim = num_gdf * 2 + num_signatures
        self.geom_mlp = nn.Sequential(
            nn.Linear(geom_input_dim, h_dim // 2),
            nn.Dropout(dropout),
            nn.BatchNorm1d(h_dim // 2),
            nn.SiLU(),
            nn.Linear(h_dim // 2, h_dim // 2),
            nn.Dropout(dropout),
            nn.BatchNorm1d(h_dim // 2),
        )

        # chem + geom feats
        self.feat_mlp = nn.Sequential(
            nn.Linear(h_dim + h_dim // 2, h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(h_dim),
        )

    def forward(self, chem_feats, geom_feats, nbr_vids):
        # chemical features
        h_chem = self.chem_mlp(chem_feats)

        # self-filter
        nbr_filter, nbr_core = h_chem.chunk(2, dim=-1)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus(nbr_core)
        h_chem = nbr_filter * nbr_core
        h_chem = scatter(h_chem, nbr_vids, dim=0, reduce="sum")

        # geometric features
        h_geom = self.geom_mlp(geom_feats)

        # combine chemical and geometric features
        h_out = self.feat_mlp(torch.cat((h_chem, h_geom), dim=-1))

        return h_out
