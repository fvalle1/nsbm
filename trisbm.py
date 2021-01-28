import graph_tool.all as gt
import numpy as np
import cloudpickle as pickle


class trisbm():
    def __init__(self):
        self.g = gt.Graph(directed=False)
        self.state = None
        
    def save_graph(self, filename="graph.xml.gz"):
        self.g.save(filename)
        
    def load_graph(self, filename="graph.xml.gz"):
        self.g = gt.load_graph(filename)
        
    def fit(self, n_init = 5, verbose=True):

        
        clabel = self.g.vp['kind']
        state_args = {'clabel': clabel, 'pclabel': clabel}
        state_args["eweight"] = self.g.ep.count
        min_entropy = np.inf
        best_state = None
        for _ in range(n_init):
            state = gt.minimize_nested_blockmodel_dl(self.g, 
                                    deg_corr=True,
                                    overlap=False,
                                    state_args=state_args,
                                    mcmc_args={'sequential': False},
                                    mcmc_equilibrate_args={'mcmc_args': {'sequential': False}},
                                    mcmc_multilevel_args={
                                          'mcmc_equilibrate_args': {
                                              'mcmc_args': {'sequential': False}
                                          },
                                          'anneal_args': {
                                              'mcmc_equilibrate_args': {
                                                   'mcmc_args': {'sequential': False}
                                              }
                                          }
                                      },
                                    verbose=verbose)
            
            entropy = state.entropy()
            if entropy < min_entropy:
                min_entropy = entropy
                self.state = state
                
    def get_groups(self, l=0):

        state_l = state.project_level(l).copy(overlap=True)
        state_l_edges = state_l.get_edge_blocks()
        B = state_l.B
        D = int(np.sum(self.g.vp['kind'].a == 0)) #documents
        W = int(np.sum(self.g.vp['kind'].a == 1)) #words
        K = int(np.sum(self.g.vp['kind'].a == 2)) #keywords

        n_dbw = np.zeros((D, B))
        n_dbw_key = np.zeros((D, B))

        for e in g.edges():
            z1, z2 = state_l_edges[e]
            v1 = e.source()
            v2 = e.target()
            if v2 < D + W:
                n_dbw[int(v1), z2] += 1
            else:
                n_dbw_key[int(v1), z2] += 1

        #p_w = np.sum(n_wb, axis=1) / float(np.sum(n_wb))

        #ind_d = np.where(np.sum(n_db, axis=0) > 0)[0]
        #Bd = len(ind_d)
        #n_db = n_db[:, ind_d]

        #ind_w = np.where(np.sum(n_wb, axis=0) > 0)[0]
        #Bw = len(ind_w)
        #n_wb = n_wb[:, ind_w]

        ind_w2 = np.where(np.sum(n_dbw, axis=0) > 0)[0]
        n_dbw = n_dbw[:, ind_w2]

        ## Mixture of word-groups into documetns P(t_w | d)
        p_tw_d = (n_dbw / np.sum(n_dbw, axis=1)[:, np.newaxis]).T

        ind_w2_keyword = np.where(np.sum(n_dbw_key, axis=0) > 0)[0]
        n_dbw_key = n_dbw_key[:, ind_w2_keyword]

        ## Mixture of word-groups into documetns P(t_w | d)
        p_tw_d = (n_dbw / np.sum(n_dbw, axis=1)[:, np.newaxis]).T

        ## Mixture of word-groups into documetns P(t_w | d)
        p_tk_d = (n_dbw_key / np.sum(n_dbw_key, axis=1)[:, np.newaxis]).T

        result = {}
        result['p_tw_d'] = p_tw_d
        result['p_tk_d'] = p_tk_d

        return result

    def draw(self, output=None) -> None:
        self.state.draw(subsample_edges = 5000, edge_pen_width = self.g.ep["count"], output=output)

    def dump_model(self, filename="trisbm.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)