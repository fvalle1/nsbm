"""
This module is cloned from https://github.com/martingerlach/hSBM_Topicmodel/commit/261d870cfc884c4f23ddaa213d07ccbddf348c78


Copyright(C) 2020 martingerlach

This program is free software: you can redistribute it and / or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY
without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see < http: // www.gnu.org/licenses/>.
"""

from __future__ import print_function
import pandas as pd
import numpy as np
import os
import sys
import argparse
from collections import Counter, defaultdict
import pickle
import graph_tool.all as gt
import sys

import scipy
from matplotlib import pyplot as plt


class sbmtm():
    '''
    Class for topic-modeling with sbm's.
    '''

    def __init__(self):
        self.g = None  # network

        self.words = []  # list of word nodes
        self.documents = []  # list of document nodes

        self.state = None  # inference state from graphtool
        self.groups = {}  # results of group membership from inference
        self.mdl = np.nan  # minimum description length of inferred state
        self.L = np.nan  # number of levels in hierarchy

    def make_graph(self, list_texts, documents=None, counts=True, n_min=None):
        '''
        Load a corpus and generate the word-document network

        optional arguments:

        :param documents: list of str, titles of documents
        :param counts: save edge-multiplicity as counts (default: True)
        :param n_min: int filter all word-nodes with less than n_min counts (default None)
        '''
        D = len(list_texts)

        # if there are no document titles, we assign integers 0,...,D-1
        # otherwise we use supplied titles
        if documents is None:
            list_titles = [str(h) for h in range(D)]
        else:
            list_titles = documents

        # make a graph
        # create a graph
        g = gt.Graph(directed=False)
        # define node properties
        # name: docs - title, words - 'word'
        # kind: docs - 0, words - 1
        name = g.vp["name"] = g.new_vp("string")
        kind = g.vp["kind"] = g.new_vp("int")
        if counts:
            ecount = g.ep["count"] = g.new_ep("int")

        docs_add = defaultdict(lambda: g.add_vertex())
        words_add = defaultdict(lambda: g.add_vertex())

        # add all documents first
        for i_d in range(D):
            title = list_titles[i_d]
            d = docs_add[title]

        # add all documents and words as nodes
        # add all tokens as links
        for i_d in range(D):
            title = list_titles[i_d]
            text = list_texts[i_d]

            d = docs_add[title]
            name[d] = title
            kind[d] = 0
            c = Counter(text)
            for word, count in c.items():
                w = words_add[word]
                name[w] = word
                kind[w] = 1
                if counts:
                    e = g.add_edge(d, w)
                    ecount[e] = count
                else:
                    for n in range(count):
                        g.add_edge(d, w)

        # filter word-types with less than n_min counts
        if n_min is not None:
            v_n = g.new_vertex_property("int")
            for v in g.vertices():
                v_n[v] = v.out_degree()

            v_filter = g.new_vertex_property("bool")
            for v in g.vertices():
                if v_n[v] < n_min and g.vp['kind'][v] == 1:
                    v_filter[v] = False
                else:
                    v_filter[v] = True
            g.set_vertex_filter(v_filter)
            g.purge_vertices()
            g.clear_filters()

        self.g = g
        self.words = [g.vp['name'][v]
                      for v in g.vertices() if g.vp['kind'][v] == 1]
        self.documents = [g.vp['name'][v]
                          for v in g.vertices() if g.vp['kind'][v] == 0]

    def make_graph_from_BoW_df(self, df, counts=True, n_min=None):
        """
        Load a graph from a Bag of Words DataFrame

        :param df: DataFrame should be a DataFrame with where df.index is a list of words and df.columns a list of documents
        :param counts: save edge-multiplicity as counts (default: True)
        :param n_min: filter all word-nodes with less than n_min counts (default None)

        """
        # make a graph
        g = gt.Graph(directed=False)
        # define node properties
        # name: docs - title, words - 'word'
        # kind: docs - 0, words - 1
        name = g.vp["name"] = g.new_vp("string")
        kind = g.vp["kind"] = g.new_vp("int")
        if counts:
            ecount = g.ep["count"] = g.new_ep("int")

        X = df.values

        # add all documents and words as nodes
        # add all tokens as links
        X = scipy.sparse.coo_matrix(X)

        if not counts and X.dtype != int:
            X_int = X.astype(int)
            if not np.allclose(X.data, X_int.data):
                raise ValueError('Data must be integer if '
                                 'weighted_edges=False')
            X = X_int

        docs_add = defaultdict(lambda: g.add_vertex())
        words_add = defaultdict(lambda: g.add_vertex())

        D = len(df.columns)
        # add all documents first
        for i_d in range(D):
            title = df.columns[i_d]
            d = docs_add[title]
            name[d] = title
            kind[d] = 0

        # add all words
        for i_d in range(len(df.index)):
            word = df.index[i_d]
            w = words_add[word]
            name[w] = word
            kind[w] = 1

        # add all documents and words as nodes
        # add all tokens as links
        for i_d in range(D):
            title = df.columns[i_d]
            text = df[title]
            for i_w, word, count in zip(range(len(df.index)), df.index, text):
                if count < 1:
                    continue
                if counts:
                    e = g.add_edge(i_d, D + i_w, add_missing=False)
                    ecount[e] = count
                else:
                    for n in range(count):
                        g.add_edge(i_d, D + i_w, add_missing=False)

        # filter word-types with less than n_min counts
        if n_min is not None:
            v_n = g.new_vertex_property("int")
            for v in g.vertices():
                v_n[v] = v.out_degree()

            v_filter = g.new_vertex_property("bool")
            for v in g.vertices():
                if v_n[v] < n_min and g.vp['kind'][v] == 1:
                    v_filter[v] = False
                else:
                    v_filter[v] = True
            g.set_vertex_filter(v_filter)
            g.purge_vertices()
            g.clear_filters()

        self.g = g
        self.words = [g.vp['name'][v]
                      for v in g.vertices() if g.vp['kind'][v] == 1]
        self.documents = [g.vp['name'][v]
                          for v in g.vertices() if g.vp['kind'][v] == 0]
        return self

    def save_graph(self, filename='graph.gt.gz'):
        '''
        Save the word-document network generated by make_graph() as filename.
        Allows for loading the graph without calling make_graph().
        '''
        self.g.save(filename)

    def load_graph(self, filename='graph.gt.gz'):
        '''
        Load a word-document network generated by make_graph() and saved with save_graph().
        '''
        self.g = gt.load_graph(filename)
        self.words = [self.g.vp['name'][v]
                      for v in self.g.vertices() if self.g.vp['kind'][v] == 1]
        self.documents = [self.g.vp['name'][v]
                          for v in self.g.vertices() if self.g.vp['kind'][v] == 0]

    def dump_model(self, filename="topsbm.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def load_model(self, filename="topsbm.pkl"):
        if self.g is not None:
            del self.g
        del self.words
        del self.documents
        if self.state is not None:
            del self.state
        del self.groups
        del self.mdl
        del self.L
        with open(filename, 'rb') as f:
            self = pickle.load(f)

    def fit(
            self,
            overlap=False,
            hierarchical=True,
            B_min=2,
            B_max=None,
            n_init=1,
            parallel=False,
            verbose=False):
        '''
        Fit the sbm to the word-document network.

        :param overlap: bool (default: False). Overlapping or Non-overlapping groups. Overlapping implemented in fit_overlap
        :param hierarchical: bool (default: True). Hierarchical SBM or Flat SBM. Flat SBM not implemented yet.
        :param Bmin: int (default:None): pass an option to the graph-tool inference specifying the minimum number of blocks.
        :param n_init: int (default:1): number of different initial conditions to run in order to avoid local minimum of MDL.
        :param parallel: passed to mcmc_sweep If parallel == False each vertex move attempt is made sequentially, where vertices are visited in random order. Otherwise the moves are attempted by sampling vertices randomly, so that the same vertex can be moved more than once, before other vertices had the chance to move.
        '''

        sequential = not parallel

        g = self.g
        if g is None:
            print('No data to fit the SBM. Load some data first (make_graph)')
        else:
            if overlap and "count" in g.ep:
                raise ValueError(
                    "When using overlapping SBMs, the graph must be constructed with 'counts=False'")
            clabel = g.vp['kind']

            state_args = {'clabel': clabel, 'pclabel': clabel}
            if "count" in g.ep:
                state_args["eweight"] = g.ep.count

            state_args["deg_corr"] = True
            state_args["overlap"] = overlap

            if B_max is None:
                B_max = self.g.num_vertices()

            # the inference
            mdl = np.inf
            for i_n_init in range(n_init):
                state_tmp = gt.minimize_nested_blockmodel_dl(
                    g, state_args=state_args, multilevel_mcmc_args={
                        "B_min": B_min, "B_max": B_max, "verbose": verbose}, )

                mdl_tmp = state_tmp.entropy()
                if mdl_tmp < mdl:
                    mdl = 1.0 * mdl_tmp
                    state = state_tmp.copy()

            self.mdl = mdl
            self.state = state
            # minimum description length
            self.mdl = self.state.entropy()
            # collect group membership for each level in the hierarchy
            L = len(state.levels)
            dict_groups_L = {}

            # only trivial bipartite structure
            if L == 2:
                self.L = 1
                for l in range(L - 1):
                    dict_groups_l = self.get_groups(l=l)
                    dict_groups_L[l] = dict_groups_l
            # omit trivial levels: l=L-1 (single group), l=L-2 (bipartite)
            else:
                self.L = L - 2
                for l in range(L - 2):
                    dict_groups_l = self.get_groups(l=l)
                    dict_groups_L[l] = dict_groups_l
            self.groups = dict_groups_L

    def fit_overlap(
            self,
            n_init=1,
            hierarchical=True,
            B_min=20,
            B_max=160,
            parallel=True,
            verbose=True):
        '''
        Fit the sbm to the word-document network.

        :param hierarchical: bool (default: True). Hierarchical SBM or Flat SBM. Flat SBM not implemented yet.
        :param Bmin: int (default:20): pass an option to the graph-tool inference specifying the minimum number of blocks.
        '''
        sequential = not parallel
        g = self.g
        clabel = g.vp['kind']
        state_args = {'clabel': clabel, 'pclabel': clabel}
        if "count" in g.ep:
            state_args["eweight"] = g.ep.count

        self.state = gt.minimize_nested_blockmodel_dl(g,
                                                      B_min=B_min,
                                                      B_max=B_max,
                                                      overlap=True,
                                                      mcmc_args={
                                                          'sequential': sequential},
                                                      mcmc_equilibrate_args={
                                                          'mcmc_args': {'sequential': sequential}},
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
                                                      nonoverlap_init=False,
                                                      deg_corr=True)
        self.mdl = self.state.entropy()
        L = len(self.state.levels)
        dict_groups_L = {}
        if L == 2:
            self.L = 1
            for l in range(L - 1):
                dict_groups_l = self.get_groups(l=l)
                dict_groups_L[l] = dict_groups_l
                # omit trivial levels: l=L-1 (single group), l=L-2 (bipartite)
        else:
            self.L = L - 2
            for l in range(L - 2):
                dict_groups_l = self.get_groups(l=l)
                dict_groups_L[l] = dict_groups_l
        self.groups = dict_groups_L

    def multiflip_mcmc_sweep(
            self,
            n_steps=1000,
            beta=np.inf,
            niter=10,
            verbose=True):
        '''
        Fit the sbm to the word-document network. Use multtiplip_mcmc_sweep
        - n_steps, int (default:1): number of steps.
        '''
        g = self.g
        if g is None:
            print('No data to fit the SBM. Load some data first (make_graph)')
        else:
            clabel = g.vp['kind']

            state_args = {'clabel': clabel, 'pclabel': clabel}
            if "count" in g.ep:
                state_args["eweight"] = g.ep.count

        state = self.state
        if state is not None:
            state = state.copy(bs=state.get_bs() +
                               [np.zeros(1)] * 4, sampling=True)
        else:
            state = gt.NestedBlockState(g)

        for step in range(n_steps):  # this should be sufficiently large
            if verbose:
                print(f"step: {step}")
            state.multiflip_mcmc_sweep(beta=beta, niter=niter)

        self.state = state
        # minimum description length
        self.mdl = self.state.entropy()
        # collect group membership for each level in the hierarchy
        L = len(state.levels)
        dict_groups_L = {}

        # only trivial bipartite structure
        if L == 2:
            self.L = 1
            for l in range(L - 1):
                dict_groups_l = self.get_groups(l=l)
                dict_groups_L[l] = dict_groups_l
        # omit trivial levels: l=L-1 (single group), l=L-2 (bipartite)
        else:
            self.L = L - 2
            for l in range(L - 2):
                dict_groups_l = self.get_groups(l=l)
                dict_groups_L[l] = dict_groups_l
        self.groups = dict_groups_L

    def multiflip_mcmc_sweep(
            self,
            n_steps=1000,
            beta=np.inf,
            niter=10,
            verbose=True):
        '''
        Fit the sbm to the word-document network. Use multtiplip_mcmc_sweep

        :param n_steps: int (default:1): number of steps.
        '''
        g = self.g
        if g is None:
            print('No data to fit the SBM. Load some data first (make_graph)')
        else:
            clabel = g.vp['kind']

            state_args = {'clabel': clabel, 'pclabel': clabel}
            if "count" in g.ep:
                state_args["eweight"] = g.ep.count

        state = self.state
        if state is not None:
            state = state.copy(bs=state.get_bs() +
                               [np.zeros(1)] * 4, sampling=True)
        else:
            state = gt.NestedBlockState(g)

        for step in range(n_steps):  # this should be sufficiently large
            if verbose:
                print(f"step: {step}")
            state.multiflip_mcmc_sweep(beta=beta, niter=niter)

        self.state = state
        # minimum description length
        self.mdl = self.state.entropy()
        # collect group membership for each level in the hierarchy
        L = len(state.levels)
        dict_groups_L = {}

        # only trivial bipartite structure
        if L == 2:
            self.L = 1
            for l in range(L - 1):
                dict_groups_l = self.get_groups(l=l)
                dict_groups_L[l] = dict_groups_l
        # omit trivial levels: l=L-1 (single group), l=L-2 (bipartite)
        else:
            self.L = L - 2
            for l in range(L - 2):
                dict_groups_l = self.get_groups(l=l)
                dict_groups_L[l] = dict_groups_l
        self.groups = dict_groups_L

    def plot(self, filename=None, nedges=1000):
        '''
        Plot the graph and group structure.

        :param filename: str; where to save the plot. if None, will not be saved
        :param nedges: int; subsample  to plot (faster, less memory)
        '''
        self.state.draw(layout='bipartite', output=filename,
                        subsample_edges=nedges, hshortcuts=1, hide=0)

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
            import io
            output = io.StringIO()
            orig_stdout = sys.stdout
            sys.stdout = output
            self.state.print_summary()
            sys.stdout = orig_stdout
            summary_text = output.getvalue()
            output.close()
            return summary_text

    def topics(self, l=0, n=10):
        '''
        get the n most common words for each word-group in level l.
        return tuples (word,P(w|tw))
        '''
        dict_groups = self.get_groups(l)
        Bw = dict_groups['Bw']
        p_w_tw = dict_groups['p_w_tw']

        words = self.words

        # loop over all word-groups
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
        dict_groups = self.get_groups(l)
        p_tw_d = dict_groups['p_tw_d']
        list_topics_tw = []
        for tw, p_tw in enumerate(p_tw_d[:, doc_index]):
            list_topics_tw += [(tw, p_tw)]
        return list_topics_tw

    def clusters(self, l=0, n=10):
        '''
        Get n 'most common' documents from each document cluster.
        most common refers to largest contribution in group membership vector.
        For the non-overlapping case, each document belongs to one and only one group with prob 1.

        '''
        dict_groups = self.get_groups(l)
        Bd = dict_groups['Bd']
        p_td_d = dict_groups['p_td_d']

        docs = self.documents
        # loop over all word-groups
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

    def clusters_query(self, doc_index, l=0):
        '''
        Get all documents in the same group as the query-document.
        Note: Works only for non-overlapping model.
        For overlapping case, we need something else.
        '''
        dict_groups = self.get_groups(l)
        Bd = dict_groups['Bd']
        p_td_d = dict_groups['p_td_d']

        documents = self.documents
        # loop over all word-groups
        dict_group_docs = {}
        td = np.argmax(p_td_d[:, doc_index])

        list_doc_index_sel = np.where(p_td_d[td, :] == 1)[0]

        list_doc_query = []

        for doc_index_sel in list_doc_index_sel:
            if doc_index != doc_index_sel:
                list_doc_query += [(doc_index_sel, documents[doc_index_sel])]

        return list_doc_query

    def group_membership(self, l=0):
        '''
        Return the group-membership vectors for
            - document-nodes, p_td_d, array with shape Bd x D
            - word-nodes, p_tw_w, array with shape Bw x V

        It gives the probability of a nodes belonging to one of the groups.
        '''
        dict_groups = self.get_groups(l)
        p_tw_w = dict_groups['p_tw_w']
        p_td_d = dict_groups['p_td_d']
        return p_td_d, p_tw_w

    def print_topics(self, l=0, format='csv', path_save='')->dict|None:
        '''
        Print topics, topic-distributions, and document clusters for a given level in the hierarchy.

        :param format: csv (default) or html
        '''
        V = self.get_V()
        D = self.get_D()

        # topics
        dict_topics = self.topics(l=l, n=-1)

        list_topics = sorted(list(dict_topics.keys()))
        list_columns = ['Topic %s' % (t + 1) for t in list_topics]

        T = len(list_topics)
        df = pd.DataFrame(columns=list_columns, index=range(V))
        if format == 'pandas':
            to_return = {}

        for t in list_topics:
            list_w = [h[0] for h in dict_topics[t]]
            V_t = len(list_w)
            df.iloc[:V_t, t] = list_w
        df = df.dropna(how='all', axis=0)
        if format == 'csv':
            fname_save = 'topsbm_level_%s_topics.csv' % (l)
            filename = os.path.join(path_save, fname_save)
            df.to_csv(filename, index=False, na_rep='')
        elif format == 'html':
            fname_save = 'topsbm_level_%s_topics.html' % (l)
            filename = os.path.join(path_save, fname_save)
            df.to_html(filename, index=False, na_rep='')
        elif format == 'tsv':
            fname_save = 'topsbm_level_%s_topics.tsv' % (l)
            filename = os.path.join(path_save, fname_save)
            df.to_csv(filename, index=False, na_rep='', sep='\t')
        elif format == 'pandas':
            to_return.update({'topsbm_level_%s_topics' % (l): df.copy()})
        else:
            pass
        

        # topic distributions
        list_columns = ['i_doc', 'doc'] + \
            ['Topic %s' % (t + 1) for t in list_topics]
        df = pd.DataFrame(columns=list_columns, index=range(D))
        for i_doc in range(D):
            list_topicdist = self.topicdist(i_doc, l=l)
            df.iloc[i_doc, 0] = i_doc
            df.iloc[i_doc, 1] = self.documents[i_doc]
            df.iloc[i_doc, 2:] = [h[1] for h in list_topicdist]
        df = df.dropna(how='all', axis=1)
        if format == 'csv':
            fname_save = 'topsbm_level_%s_topic-dist.csv' % (l)
            filename = os.path.join(path_save, fname_save)
            df.to_csv(filename, index=False, na_rep='')
        elif format == 'html':
            fname_save = 'topsbm_level_%s_topic-dist.html' % (l)
            filename = os.path.join(path_save, fname_save)
            df.to_html(filename, index=False, na_rep='')
        elif format == 'pandas':
            to_return.update({'topsbm_level_%s_topic-dist' % (l): df.copy()})
        else:
            pass

        # doc-groups

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
            fname_save = 'topsbm_level_%s_clusters.csv' % (l)
            filename = os.path.join(path_save, fname_save)
            df.to_csv(filename, index=False, na_rep='')
        elif format == 'html':
            fname_save = 'topsbm_level_%s_clusters.html' % (l)
            filename = os.path.join(path_save, fname_save)
            df.to_html(filename, index=False, na_rep='')
        elif format == 'pandas':
            to_return.update({'topsbm_level_%s_clusters' % (l): df.copy()})
        else:
            pass

        # word-distr
        list_topics = np.arange(len(self.get_groups(l)['p_w_tw'].T))
        list_columns = ["Topic %d" % (t + 1) for t in list_topics]

        pwtw_df = pd.DataFrame(data=self.get_groups(
            l)['p_w_tw'], index=self.words, columns=list_columns)
        pwtw_df.replace(0, np.nan)
        pwtw_df = pwtw_df.dropna(how='all', axis=0)
        pwtw_df.replace(np.nan, 0)
        if format == 'csv':
            fname_save = "topsbm_level_%d_word-dist.csv" % l
            filename = os.path.join(path_save, fname_save)
            pwtw_df.to_csv(filename, index=True, header=True, na_rep='')
        elif format == 'html':
            fname_save = "topsbm_level_%d_word-dist.html" % l
            filename = os.path.join(path_save, fname_save)
            pwtw_df.to_html(filename, index=True, na_rep='')
        elif format == 'pandas':
            to_return.update({'topsbm_level_%d_word-dist' % (l): pwtw_df.copy()})
        else:
            pass

        if format == 'pandas':
            return to_return

    ###########
    # HELPER FUNCTIONS
    ###########
    def get_mdl(self):
        return self.mdl

    # get group-topic statistics
    def get_groups(self, l=0):
        '''
        extract statistics on group membership of nodes form the inferred state.

        :param B_d: int, number of doc-groups
        :param B_w: int, number of word-groups
        :param p_tw_w: array B_w x V; word-group-membership: prob that word-node w belongs to word-group tw: P(tw | w)
        :param p_td_d: array B_d x D; doc-group membership: prob that doc-node d belongs to doc-group td: P(td | d)
        :param p_w_tw: array V x B_w; topic distribution: prob of word w given topic tw P(w | tw)
        :param p_tw_d: array B_w x d; doc-topic mixtures: prob of word-group tw in doc d P(tw | d)

        :return: dictionary

        '''
        V = self.get_V()
        D = self.get_D()
        N = self.get_N()

        if l in self.groups.keys():
            return self.groups[l]

        g = self.g
        state = self.state
        state_l = state.project_level(l).copy(overlap=True)
        state_l_edges = state_l.get_edge_blocks()  # labeled half-edges

        counts = 'count' in self.g.ep.keys()

        # count labeled half-edges, group-memberships
        B = state_l.get_B()
        # number of half-edges incident on word-node w and labeled as
        # word-group tw
        n_wb = np.zeros((V, B))
        # number of half-edges incident on document-node d and labeled as
        # document-group td
        n_db = np.zeros((D, B))
        # number of half-edges incident on document-node d and labeled as
        # word-group td
        n_dbw = np.zeros((D, B))

        for e in g.edges():
            z1, z2 = state_l_edges[e]
            v1 = e.source()
            v2 = e.target()
            if counts:
                weight = g.ep["count"][e]
            else:
                weight = 1
            n_db[int(v1), z1] += weight
            n_dbw[int(v1), z2] += weight
            n_wb[int(v2) - D, z2] += weight

        p_w = np.sum(n_wb, axis=1) / float(np.sum(n_wb))

        ind_d = np.where(np.sum(n_db, axis=0) > 0)[0]
        Bd = len(ind_d)
        n_db = n_db[:, ind_d]

        ind_w = np.where(np.sum(n_wb, axis=0) > 0)[0]
        Bw = len(ind_w)
        n_wb = n_wb[:, ind_w]

        ind_w2 = np.where(np.sum(n_dbw, axis=0) > 0)[0]
        n_dbw = n_dbw[:, ind_w2]

        # group-membership distributions
        # group membership of each word-node P(t_w | w)
        p_tw_w = (n_wb / np.sum(n_wb, axis=1)[:, np.newaxis]).T

        # group membership of each doc-node P(t_d | d)
        p_td_d = (n_db / np.sum(n_db, axis=1)[:, np.newaxis]).T

        # topic-distribution for words P(w | t_w)
        p_w_tw = n_wb / np.sum(n_wb, axis=0)[np.newaxis, :]

        # Mixture of word-groups into documetns P(t_w | d)
        p_tw_d = (n_dbw / np.sum(n_dbw, axis=1)[:, np.newaxis]).T

        result = {}
        result['Bd'] = Bd
        result['Bw'] = Bw
        result['p_tw_w'] = p_tw_w
        result['p_td_d'] = p_td_d
        result['p_w_tw'] = p_w_tw
        result['p_tw_d'] = p_tw_d

        self.groups[l] = result

        return result

    def search_consensus(self, force_niter=100000, niter=100):
        # collect nested partitions
        bs = []

        def collect_partitions(s):
            bs.append(s.get_bs())

        # Now we collect the marginals for exactly niter sweeps
        gt.mcmc_equilibrate(
            self.state,
            force_niter=force_niter,
            mcmc_args=dict(
                niter=niter),
            callback=collect_partitions)

        # Disambiguate partitions and obtain marginals
        pmode = gt.PartitionModeState(bs, nested=True, converge=True)
        pv = pmode.get_marginal(self.g)

        # Get consensus estimate
        bs = pmode.get_max_nested()
        self.state = self.state.copy(bs=bs)

        return pv

    # helper functions

    def get_V(self):
        '''
        :return: number of word-nodes == types
        '''
        return int(np.sum(self.g.vp['kind'].a == 1))  # no. of types

    def get_D(self):
        '''
        :return: number of doc-nodes == number of documents
        '''
        return int(np.sum(self.g.vp['kind'].a == 0))  # no. of types

    def get_N(self):
        '''
        :return: number of edges == tokens
        '''
        return int(self.g.num_edges())  # no. of types

    def group_to_group_mixture(self, l=0, norm=True):
        V = self.get_V()
        D = self.get_D()
        N = self.get_N()

        g = self.g
        state = self.state
        state_l = state.project_level(l).copy(overlap=True)
        state_l_edges = state_l.get_edge_blocks()  # labeled half-edges

        # count labeled half-edges, group-memberships
        B = state_l.get_B()
        n_td_tw = np.zeros((B, B))

        counts = 'count' in self.g.ep.keys()

        for e in g.edges():
            z1, z2 = state_l_edges[e]
            if counts:
                n_td_tw[z1, z2] += g.ep["count"][e]
            else:
                n_td_tw[z1, z2] += 1

        ind_d = np.where(np.sum(n_td_tw, axis=1) > 0)[0]
        Bd = len(ind_d)
        ind_w = np.where(np.sum(n_td_tw, axis=0) > 0)[0]
        Bw = len(ind_w)

        n_td_tw = n_td_tw[:Bd, Bd:]
        if norm:
            return n_td_tw / np.sum(n_td_tw)
        else:
            return n_td_tw

    def plot_topic_dist(self, l):
        groups = self.get_groups(l)
        p_w_tw = groups['p_w_tw']
        fig = plt.figure(figsize=(12, 10))
        plt.imshow(p_w_tw, origin='lower', aspect='auto', interpolation='none')
        plt.title(r'Word group membership $P(w | tw)$')
        plt.xlabel('Topic, tw')
        plt.ylabel('Word w (index)')
        plt.colorbar()
        fig.savefig("p_w_tw_%d.png" % l)
        p_tw_d = groups['p_tw_d']
        fig = plt.figure(figsize=(12, 10))
        plt.imshow(p_tw_d, origin='lower', aspect='auto', interpolation='none')
        plt.title(r'Word group membership $P(tw | d)$')
        plt.xlabel('Document (index)')
        plt.ylabel('Topic, tw')
        plt.colorbar()
        fig.savefig("p_tw_d_%d.png" % l)

    def save_data(self):
        for i in range(len(self.state.get_levels()) - 2)[::-1]:
            print("Saving level %d" % i)
            self.print_topics(l=i)
            self.print_topics(l=i, format='tsv')
            self.plot_topic_dist(i)
            e = self.state.get_levels()[i].get_matrix()
            plt.matshow(e.todense())
            plt.savefig("mat_%d.png" % i)
        self.print_summary()

    def serialize_data(self, save = False):
        data = {
            "g": self.g,
            "words": list(self.words),
            "documents": list(self.documents),
            "mdl": self.mdl,
        }
        data.update({
            "levels":
                [{"topics": self.print_topics(l=l, format='pandas'),
                  "block_matrix": self.state.get_levels()[l].get_matrix().todense(),
                  # "plot_topic_dist": self.plot_topic_dist(l)
                  }
                 for l in range(len(self.state.get_levels()) - 2)
                 ],
            "summary": self.print_summary(tofile=False)
        })
        
        if save:
            import pickle
            with open("topsbm_data.pkl", "wb") as f:
                pickle.dump(data, f)
        
        return data
    
    def save_rdata(self):
        import rpy2.robjects as ro
        import rpy2.robjects.packages as rpackages
        from rpy2.robjects import r
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects import pandas2ri, numpy2ri

        # Ensure the 'base' package is loaded
        rpackages.importr('base')
        pandas2ri.activate()
        numpy2ri.activate()
        
        def process_topic_df(df):
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = df[col].astype(float)
                    except ValueError:
                        df[col] = df[col].astype(str)
            return ro.conversion.py2rpy(df)

        # Convert Python data to R objects
        with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
            r_data = ro.ListVector({
                "words": ro.StrVector(self.words),
                "documents": ro.StrVector(self.documents),
                "mdl": ro.FloatVector([self.mdl]),  # Wrap the float in a list
                "levels": ro.ListVector([
                    (l, ro.ListVector({
                        "topics": ro.ListVector({k:process_topic_df(v) for k,v in self.print_topics(l=l, format='pandas').items()}),
                        "block_matrix": ro.conversion.py2rpy(self.state.get_levels()[l].get_matrix().toarray())
                    })) for l in range(len(self.state.get_levels()) - 2)
                ]),
                "summary": ro.StrVector([self.print_summary(tofile=False)])
            })
          
        ro.r.assign("topsbm_results", r_data)
        ro.r("save(topsbm_results, file='{}')".format("topsbm_data.RData"))
