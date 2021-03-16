import graph_tool.all as gt
import numpy as np
import pandas as pd
import cloudpickle as pickle
import os, sys
import matplotlib.pyplot as plt

class trisbm():
    def __init__(self):
        self.g = gt.Graph(directed=False)
        self.state = None
        self.mdl = np.inf
        self.documents = None
        self.words = None
        self.keywords = None
        self.groups = []
        
    def save_graph(self, filename="graph.xml.gz"):
        self.g.save(filename)
        
    def load_graph(self, filename="graph.xml.gz"):
        self.g = gt.load_graph(filename)
        self.documents = [self.g.vp['name'][v] for v in self.g.vertices() if self.g.vp['kind'][v] == 0]
        self.words = [self.g.vp['name'][v] for v in self.g.vertices() if self.g.vp['kind'][v] == 1]
        self.keywords = [self.g.vp['name'][v] for v in self.g.vertices() if self.g.vp['kind'][v] == 2]
        
        
    def make_graph(self, df, get_kind):
        """
        :param df: DataFrame with words on index and texts on columns
        :param get_kind: function that returns 1 or 2 given a word
        """
        self.g = gt.Graph(directed=False)
        name = self.g.vp["name"] = self.g.new_vp("string")
        kind = self.g.vp["kind"] = self.g.new_vp("int")
        weight = self.g.ep["count"] = self.g.new_ep("float")
        
        for doc in df.columns:
            d = self.g.add_vertex()
            name[d] = doc
            kind[d] = 0
            
        for word in df.index:
            w = self.g.add_vertex()
            name[w] = word
            kind[w] = get_kind(word)
            
        D = df.shape[1]
        
        for i_doc, doc in enumerate(df.columns):
            text = df[doc]
            self.g.add_edge_list([(i_doc,D + x[0][0],np.log2(1+x[1])) for x in zip(enumerate(df.index),text)], eprops=[weight])

        filter_edges = self.g.new_edge_property("bool")
        for e in self.g.edges():
            filter_edges[e] = weight[e]>0

        self.g.set_edge_filter(filter_edges)
        self.g.purge_edges()
        self.g.clear_filters()
        
        self.documents = df.columns
        self.words = df.index[self.g.vp['kind'].a[D:] == 1]
        self.keywords = df.index[self.g.vp['kind'].a[D:] == 2]
        
    def fit(self, n_init = 5, verbose=True, deg_corr=True, overlap=False, parallel=True, *args, **kwargs):
        """
        Fit using minimize_nested_blockmodel_dl
        
        :param n_init:
        """
        
        sequential = not parallel

        clabel = self.g.vp['kind']
        state_args = {'clabel': clabel, 'pclabel': clabel}
        state_args["eweight"] = self.g.ep.count
        min_entropy = np.inf
        best_state = None
        for _ in range(n_init):
            state = gt.minimize_nested_blockmodel_dl(self.g, 
                                    deg_corr = deg_corr,
                                    overlap = overlap,
                                    state_args=state_args,
                                    mcmc_args={'sequential': sequential},
                                    mcmc_equilibrate_args={'mcmc_args': {'sequential': sequential}},
                                    mcmc_multilevel_args={
                                          'mcmc_equilibrate_args': {
                                              'mcmc_args': {'sequential': sequential}
                                          },
                                          'anneal_args': {
                                              'mcmc_equilibrate_args': {
                                                   'mcmc_args': {'sequential': sequential}
                                              }
                                          }
                                      },
                                    verbose=verbose,
                                    *args, 
                                    **kwargs)
            
            entropy = state.entropy()
            if entropy < min_entropy:
                min_entropy = entropy
                self.state = state
                
        self.mdl = min_entropy
        
        L = len(self.state.levels)
        dict_groups_L = {}

        ## only trivial bipartite structure
        if L == 2:
            self.L = 1
            for l in range(L - 1):
                dict_groups_l = self.get_groups(l=l)
                dict_groups_L[l] = dict_groups_l
        ## omit trivial levels: l=L-1 (single group), l=L-2 (tripartite)
        else:
            self.L = L - 2
            for l in range(L - 2):
                dict_groups_l = self.get_groups(l=l)
                dict_groups_L[l] = dict_groups_l
        self.groups = dict_groups_L
                
    def get_groups(self, l=0):
        state_l = self.state.project_level(l).copy(overlap=True)
        state_l_edges = state_l.get_edge_blocks()
        B = state_l.B
        D, W, K = self._get_shape()

        n_wb = np.zeros((W, B))  ## number of half-edges incident on word-node w and labeled as word-group tw
        n_w_key_b = np.zeros((K, B))  ## number of half-edges incident on word-node w and labeled as word-group tw
        n_db = np.zeros((D, B))  ## number of half-edges incident on document-node d and labeled as document-group td
        n_dbw = np.zeros((D, B)) ## number of half-edges incident on document-node d and labeled as word-group tw
        n_dbw_key = np.zeros((D, B)) ## number of half-edges incident on document-node d and labeled as keyword-group tw_key

        for e in self.g.edges():
            z1, z2 = state_l_edges[e]
            v1 = e.source()
            v2 = e.target()
            weight = self.g.ep["count"][e]
            n_db[int(v1), z1] += weight
            if self.g.vp['kind'][v2] == 1:
                n_wb[int(v2) - D, z2] += weight
                n_dbw[int(v1), z2] += weight
            else:
                n_w_key_b[int(v2) - D - W, z2] += weight
                n_dbw_key[int(v1), z2] += weight

        p_w = np.sum(n_wb, axis=1) / float(np.sum(n_wb))
        p_w_key = np.sum(n_w_key_b, axis=1) / float(np.sum(n_w_key_b))

        ind_d = np.where(np.sum(n_db, axis=0) > 0)[0]
        Bd = len(ind_d)
        n_db = n_db[:, ind_d]

        ind_w = np.where(np.sum(n_wb, axis=0) > 0)[0]
        Bw = len(ind_w)
        n_wb = n_wb[:, ind_w]

        ind_w2 = np.where(np.sum(n_dbw, axis=0) > 0)[0]
        n_dbw = n_dbw[:, ind_w2]

        ind_w_key = np.where(np.sum(n_w_key_b, axis=0) > 0)[0]
        Bk = len(ind_w_key)
        n_w_key_b = n_w_key_b[:, ind_w_key]
        
        ind_w2_keyword = np.where(np.sum(n_dbw_key, axis=0) > 0)[0]
        n_dbw_key = n_dbw_key[:, ind_w2_keyword]
        

        # group membership of each word-node P(t_w | w)
        p_tw_w = (n_wb / np.sum(n_wb, axis=1)[:, np.newaxis]).T

        # group membership of each keyword-node P(t_k | keyword)
        p_tk_w_key = (n_w_key_b / np.sum(n_w_key_b, axis=1)[:, np.newaxis]).T
        
        ## topic-distribution for words P(w | t_w)
        p_w_tw = n_wb / np.sum(n_wb, axis=0)[np.newaxis, :]
        
        ## topickey-distribution for keywords P(keyword | t_w_key)
        p_w_key_tk = n_w_key_b / np.sum(n_w_key_b, axis=0)[np.newaxis, :]
        
        ## Mixture of word-groups into documetns P(t_w | d)
        p_tw_d = (n_dbw / np.sum(n_dbw, axis=1)[:, np.newaxis]).T

        ## Mixture of word-groups into documetns P(t_w | d)
        p_tk_d = (n_dbw_key / np.sum(n_dbw_key, axis=1)[:, np.newaxis]).T
        
        # group membership of each doc-node P(t_d | d)
        p_td_d = (n_db / np.sum(n_db, axis=1)[:, np.newaxis]).T

        result = {}
        result['Bd'] = Bd
        result['Bw'] = Bw
        result['Bk'] = Bk
        result['p_tw_w'] = p_tw_w
        result["p_tk_w_key"]=p_tk_w_key
        result['p_td_d'] = p_td_d
        result['p_w_tw'] = p_w_tw
        result['p_w_key_tk'] = p_w_key_tk
        result['p_tw_d'] = p_tw_d
        result['p_tk_d'] = p_tk_d

        return result
    
    def clusters(self, l=0, n=10):
        '''
        Get n 'most common' documents from each document cluster.
        most common refers to largest contribution in group membership vector.
        For the non-overlapping case, each document belongs to one and only one group with prob 1.

        '''
        dict_groups = self.groups[l]
        Bd = dict_groups['Bd']
        p_td_d = dict_groups['p_td_d']

        docs = self.documents
        ## loop over all word-groups
        dict_group_docs = {}
        for td in range(Bd):
            p_d_ = p_td_d[td, :]
            ind_d_ = np.argsort(p_d_)[::-1]
            list_docs_td = []
            for i in ind_d_[:n]:
                if p_d_[i] > 0:
                    list_docs_td += [(docs[i], p_d_[i])]
                else:
                    break
            dict_group_docs[td] = list_docs_td
        return dict_group_docs
    
    def topics(self, l=0, n=10):
        '''
        get the n most common words for each word-group in level l.
        return tuples (word,P(w|tw))
        '''
        dict_groups = self.groups[l]
        Bw = dict_groups['Bw']
        p_w_tw = dict_groups['p_w_tw']

        words = self.words

        ## loop over all word-groups
        dict_group_words = {}
        for tw in range(Bw):
            p_w_ = p_w_tw[:, tw]
            ind_w_ = np.argsort(p_w_)[::-1]
            list_words_tw = []
            for i in ind_w_[:n]:
                if p_w_[i] > 0:
                    list_words_tw += [(words[i], p_w_[i])]
                else:
                    break
            dict_group_words[tw] = list_words_tw
        return dict_group_words

    def topicdist(self, doc_index, l=0):
        dict_groups = self.groups[l]
        p_tw_d = dict_groups['p_tw_d']
        list_topics_tw = []
        for tw, p_tw in enumerate(p_tw_d[:, doc_index]):
            list_topics_tw += [(tw, p_tw)]
        return list_topics_tw
    
    def metadata(self, l=0, n=10):
        '''
        get the n most common keywords for each keyword-group in level l.
        return tuples (keyword,P(kw|tk))
        '''
        dict_groups = self.groups[l]
        Bw = dict_groups['Bk']
        p_w_tw = dict_groups['p_w_key_tk']

        words = self.keywords

        ## loop over all word-groups
        dict_group_keywords = {}
        for tw in range(Bw):
            p_w_ = p_w_tw[:, tw]
            ind_w_ = np.argsort(p_w_)[::-1]
            list_words_tw = []
            for i in ind_w_[:n]:
                if p_w_[i] > 0:
                    list_words_tw += [(words[i], p_w_[i])]
                else:
                    break
            dict_group_keywords[tw] = list_words_tw
        return dict_group_keywords

    def metadatumdist(self, doc_index, l=0):
        dict_groups = self.groups[l]
        p_tk_d = dict_groups['p_tk_d']
        list_topics_tk = []
        for tk, p_tk in enumerate(p_tk_d[:, doc_index]):
            list_topics_tk += [(tk, p_tk)]
        return list_topics_tk
    
    def print_topics(self, l=0, format='csv', path_save=''):
        '''
        Print topics, topic-distributions, and document clusters for a given level in the hierarchy.
        format: csv (default) or html
        '''
        D, W, K = self._get_shape()

        ## topics
        dict_topics = self.topics(l=l, n=-1)

        list_topics = sorted(list(dict_topics.keys()))
        list_columns = ['Topic %s' % (t + 1) for t in list_topics]

        T = len(list_topics)
        df = pd.DataFrame(columns=list_columns, index=range(W))

        for t in list_topics:
            list_w = [h[0] for h in dict_topics[t]]
            V_t = len(list_w)
            df.iloc[:V_t, t] = list_w
        df = df.dropna(how='all', axis=0)
        if format == 'csv':
            fname_save = 'trisbm_level_%s_topics.csv' % (l)
            filename = os.path.join(path_save, fname_save)
            df.to_csv(filename, index=False, na_rep='')
        elif format == 'html':
            fname_save = 'trisbm_level_%s_topics.html' % (l)
            filename = os.path.join(path_save, fname_save)
            df.to_html(filename, index=False, na_rep='')
        elif format == 'tsv':
            fname_save = 'trisbm_level_%s_topics.tsv' % (l)
            filename = os.path.join(path_save, fname_save)
            df.to_csv(filename, index=False, na_rep='', sep='\t')
        else:
            pass

        ## topic distributions
        list_columns = ['i_doc', 'doc'] + ['Topic %s' % (t + 1) for t in list_topics]
        df = pd.DataFrame(columns=list_columns, index=range(D))
        for i_doc in range(D):
            list_topicdist = self.topicdist(i_doc, l=l)
            df.iloc[i_doc, 0] = i_doc
            df.iloc[i_doc, 1] = self.documents[i_doc]
            df.iloc[i_doc, 2:] = [h[1] for h in list_topicdist]
        df = df.dropna(how='all', axis=1)
        if format == 'csv':
            fname_save = 'trisbm_level_%s_topic-dist.csv' % (l)
            filename = os.path.join(path_save, fname_save)
            df.to_csv(filename, index=False, na_rep='')
        elif format == 'html':
            fname_save = 'trisbm_level_%s_topic-dist.html' % (l)
            filename = os.path.join(path_save, fname_save)
            df.to_html(filename, index=False, na_rep='')
        else:
            pass
        
        ## keywords
        dict_metadata = self.metadata(l=l, n=-1)

        list_metadata = sorted(list(dict_metadata.keys()))
        list_columns = ['Metadatum %s' % (t + 1) for t in list_metadata]

        T = len(list_topics)
        df = pd.DataFrame(columns=list_columns, index=range(K))

        for t in list_metadata:
            list_w = [h[0] for h in dict_metadata[t]]
            V_t = len(list_w)
            df.iloc[:V_t, t] = list_w
        df = df.dropna(how='all', axis=0)
        if format == 'csv':
            fname_save = 'trisbm_level_%s_metadata.csv' % (l)
            filename = os.path.join(path_save, fname_save)
            df.to_csv(filename, index=False, na_rep='')
        elif format == 'html':
            fname_save = 'trisbm_level_%s_metadata.html' % (l)
            filename = os.path.join(path_save, fname_save)
            df.to_html(filename, index=False, na_rep='')
        elif format == 'tsv':
            fname_save = 'trisbm_level_%s_metadata.tsv' % (l)
            filename = os.path.join(path_save, fname_save)
            df.to_csv(filename, index=False, na_rep='', sep='\t')
        else:
            pass

        ## metadata distributions
        list_columns = ['i_doc', 'doc'] + ['Metadatum %s' % (t + 1) for t in list_metadata]
        df = pd.DataFrame(columns=list_columns, index=range(D))
        for i_doc in range(D):
            list_topicdist = self.metadatumdist(i_doc, l=l)
            df.iloc[i_doc, 0] = i_doc
            df.iloc[i_doc, 1] = self.documents[i_doc]
            df.iloc[i_doc, 2:] = [h[1] for h in list_topicdist]
        df = df.dropna(how='all', axis=1)
        if format == 'csv':
            fname_save = 'trisbm_level_%s_metadatum-dist.csv' % (l)
            filename = os.path.join(path_save, fname_save)
            df.to_csv(filename, index=False, na_rep='')
        elif format == 'html':
            fname_save = 'trisbm_level_%s_metadatum-dist.html' % (l)
            filename = os.path.join(path_save, fname_save)
            df.to_html(filename, index=False, na_rep='')
        else:
            pass

        ## doc-groups

        dict_clusters = self.clusters(l=l, n=-1)

        list_clusters = sorted(list(dict_clusters.keys()))
        list_columns = ['Cluster %s' % (t + 1) for t in list_clusters]

        T = len(list_clusters)
        df = pd.DataFrame(columns=list_columns, index=range(D))

        for t in list_clusters:
            list_d = [h[0] for h in dict_clusters[t]]
            D_t = len(list_d)
            df.iloc[:D_t, t] = list_d
        df = df.dropna(how='all', axis=0)
        if format == 'csv':
            fname_save = 'trisbm_level_%s_clusters.csv' % (l)
            filename = os.path.join(path_save, fname_save)
            df.to_csv(filename, index=False, na_rep='')
        elif format == 'html':
            fname_save = 'trisbm_level_%s_clusters.html' % (l)
            filename = os.path.join(path_save, fname_save)
            df.to_html(filename, index=False, na_rep='')
        else:
            pass

        ## word-distr
        list_topics = np.arange(len(self.groups[l]['p_w_tw'].T))
        list_columns = ["Topic %d" % (t + 1) for t in list_topics]

        pwtw_df = pd.DataFrame(data=self.groups[l]['p_w_tw'], index=self.words, columns=list_columns)
        pwtw_df.replace(0, np.nan)
        pwtw_df = pwtw_df.dropna(how='all', axis=0)
        pwtw_df.replace(np.nan, 0)
        if format == 'csv':
            fname_save = "trisbm_level_%d_word-dist.csv" % l
            filename = os.path.join(path_save, fname_save)
            pwtw_df.to_csv(filename, index=True, header=True, na_rep='')
        elif format == 'html':
            fname_save = "trisbm_level_%d_word-dist.html" % l
            filename = os.path.join(path_save, fname_save)
            pwtw_df.to_html(filename, index=True, na_rep='')
        else:
            pass
        
        
        ## keyword-distr
        list_topics = np.arange(len(self.groups[l]['p_w_key_tk'].T))
        list_columns = ["Metadatum %d" % (t + 1) for t in list_topics]

        pw_key_tk_df = pd.DataFrame(data=self.groups[l]['p_w_key_tk'], index=self.keywords, columns=list_columns)
        pw_key_tk_df.replace(0, np.nan)
        pw_key_tk_df = pw_key_tk_df.dropna(how='all', axis=0)
        pw_key_tk_df.replace(np.nan, 0)
        if format == 'csv':
            fname_save = "trisbm_level_%d_keyword-dist.csv" % l
            filename = os.path.join(path_save, fname_save)
            pw_key_tk_df.to_csv(filename, index=True, header=True, na_rep='')
        elif format == 'html':
            fname_save = "trisbm_level_%d_keyword-dist.html" % l
            filename = os.path.join(path_save, fname_save)
            pw_key_tk_df.to_html(filename, index=True, na_rep='')
        else:
            pass

    def draw(self, **kwargs) -> None:
        self.state.draw(subsample_edges = 5000, edge_pen_width = self.g.ep["count"], **kwargs)

    def dump_model(self, filename="trisbm.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
            
            
            
    def print_summary(self, tofile=True):
        '''
        Print hierarchy summary
        '''
        if tofile:
            orig_stdout = sys.stdout
            f = open('summary.txt', 'w')
            sys.stdout = f
            self.state.print_summary()
            sys.stdout = orig_stdout
            f.close()
        else:
            self.state.print_summary()
            
    def save_data(self):
        for i in range(len(self.state.get_levels()) - 2)[::-1]:
            print("Saving level %d" % i)
            self.print_topics(l=i)
            self.print_topics(l=i, format='tsv')
            e = self.state.get_levels()[i].get_matrix()
            plt.matshow(e.todense())
            plt.savefig("mat_%d.png" % i)
        self.print_summary()
            
    def _get_shape(self):
        """
        :return: tuple (number of documents, number of words, number of keywords)
        """
        D = int(np.sum(self.g.vp['kind'].a == 0)) #documents
        W = int(np.sum(self.g.vp['kind'].a == 1)) #words
        K = int(np.sum(self.g.vp['kind'].a == 2)) #keywords
        return D, W, K