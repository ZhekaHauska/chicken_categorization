import seaborn as sns
from hmmlearn.hmm import CategoricalHMM
from tqdm import tqdm
from collections import Counter, OrderedDict
from math import floor
import numpy as np
import pandas as pd


class HMMFitter:
    def __init__(
        self,
        sequences
    ):
        """
        X: array-like flattened chain array
        lengths: array of lengths of chains
        """
        self.X = sequences
        self.lock_emission = None
        self.components_per_state = None
        self.model = None

    def fit_model(
        self,
        n_components = (15, 30, 50, 100),
        filter_colors = None,
        lock_emission = False,
        account_frequencies = False,
        max_cells_per_column = 10,
        min_cells_per_column = 1,
        max_iter=10,
        tol=1e-2,
        verbose=True,
        seeds = (43, 3423, 135139, 434231, 32222)
    ):
        """
        model is created fitted and now can be used
        statistics of training saved

        model_samples_num = 10: number of times we refit the model with data
        n_components = None: array of number of components of the model
        loc_emission: boolean if checked True we set the emission matrix
        account_frequencies: boolean if checked states for observations are distributed proportionally to frequencies of these observations
        ___________
        returns:
        nothing
        """
        if filter_colors is not None:
            seqs = []
            lengths = []
            for seq in self.X:
                filtered = seq[np.isin(seq, filter_colors, invert=True)]
                seqs.append(filtered)
                lengths.append(len(filtered))
            X = np.concat(seqs)[:, None]
            # remap colors to states
            colors = np.unique(X)
            mapping = np.zeros(colors.max()+1, dtype=np.int32)
            mapping[colors] = np.arange(colors.size)
            X = mapping[X.flatten()][:, None]
        else:
            X = self.X.flatten()[:, None]
            lengths = [self.X.shape[1]] * self.X.shape[0]

        n_possible_obs = np.unique(X).size
        print(f'total data points: {len(X)}, obs states: {n_possible_obs}')

        components_per_state = None
        if account_frequencies:
            if not lock_emission:
                print("account_frequencies is set to True\nsetting lock_emission to True")
            self.lock_emission = True
        else:
            self.lock_emission = lock_emission

        best_score = best_model = None

        model_scores = {
            'idx': [],
            'n_comp': [],
            'score': [],
            }

        model_scores = pd.DataFrame(model_scores).set_index('idx')

        for n in tqdm(n_components):
            if self.lock_emission:
                if account_frequencies:
                    frequencies = OrderedDict(sorted(Counter(X.flatten()).items()))
                    components_per_state = np.array(
                        [
                            max(
                                min_cells_per_column,
                                min(
                                    max_cells_per_column, floor(n*(frequencies[i] / len(X)))
                                )
                            )
                            for i in frequencies.keys()
                        ]
                    )
                    n = components_per_state.sum()
                    print(components_per_state, n)
                    emprob = np.zeros((n, n_possible_obs))
                    diag = [[i]*(components_per_state[i]) for i in range(n_possible_obs)]
                    diag = [x for xs in diag for x in xs]
                    diag = np.array(diag).flatten()
                    for i in range(len(diag)):
                        emprob[i, diag[i]] = 1
                else:
                    emprob = np.zeros((n, n_possible_obs))
                    diag = np.array([[i]*(n//n_possible_obs) for i in range(n_possible_obs)]).flatten()
                    components_per_state = np.array([(n//n_possible_obs) for i in range(n_possible_obs)])
                    for i in range(len(diag)):
                        emprob[i, diag[i]] = 1

            for seed in seeds:
                if self.lock_emission:
                    model = CategoricalHMM(
                        n_components=n,
                        random_state=seed,
                        params='st',
                        init_params='st',
                        n_features=n_possible_obs,
                        verbose=verbose,
                        n_iter=max_iter,
                        tol=tol
                    )
                    model.emissionprob_ = emprob
                else:
                    model = CategoricalHMM(
                        n_components=n,
                        n_features=n_possible_obs,
                        random_state=seed,
                        verbose=verbose,
                        n_iter=max_iter,
                        tol=tol
                    )
                model.fit(
                    X,
                    lengths=lengths
                )
                score = model.score(X, lengths=lengths)/np.sum(lengths)
                model_scores.loc[len(model_scores)+1] = [n, score]
                if best_score is None or score > best_score:
                    best_model = model
                    best_score = score
                    self.components_per_state = components_per_state

        self.model = best_model
        self.scores = model_scores
        self.startprob_ = best_model.startprob_

    def draw_heatmap(self, matrix_type, ax, add_title = ''):
        """
        matrix_type: 'emission', 'transition' -- what type of matrix is wanted displayed
        ax: matplotlib.axes -- axes where the heatmap is plotted
        add_title: string -- additional info for title
        """
        if matrix_type == 'emission':
            sns.heatmap(self.model.emissionprob_, linewidths=0.5, linecolor='white', ax = ax)
            ax.set_title(add_title + ' emission mat')
        if matrix_type == 'transition':
            if self.lock_emission:
                sns.heatmap(self.model.transmat_, ax = ax)
                x = self.components_per_state[0]
                ax.axvline(x=x, linestyle='--', color='lightgray')
                ax.axhline(y=x, linestyle='--', color='lightgray')
                for i in self.components_per_state[1:]:
                    x+=i
                    ax.axvline(x=x, linestyle='--', color='lightgray')
                    ax.axhline(y=x, linestyle='--', color='lightgray')
            else: sns.heatmap(self.model.transmat_, linewidths=0.5, linecolor='white', ax = ax)
            ax.set_title(add_title+' transition mat')
            if self.components_per_state is not None:
                ax.set_xticks(np.cumsum(self.components_per_state))
                ax.set_yticks(np.cumsum(self.components_per_state))
                ax.set_xticklabels(sorted(np.unique(self.X)))
                ax.set_yticklabels(sorted(np.unique(self.X)))

    def bar_scores(self, ax):
        '''
        ax: matplotlib.axes
        '''
        df = pd.concat([self.scores.groupby(by = 'n_comp').mean().rename(columns={'score':'mean_score'}), self.scores.groupby(by = 'n_comp').std().rename(columns={'score':'std_score'})], axis = 1)
        df['number_components'] = df.index
        df['idx'] = np.arange(len(df))
        df.set_index('idx', inplace=True)

        barplot = sns.barplot(x='number_components', y='mean_score', data=df, color='lightblue', errorbar=None, ax = ax)

        # Add error bars
        for index, row in df.iterrows():
            ax.errorbar(x=index, y=row['mean_score'], yerr=row['std_score'], fmt='none', color='black', capsize=5)
        ax.set_title('Average scores')
        ax.set_xlabel('number of components')
        ax.set_ylabel('score')

    def predict_distr_for_all(self, obs_seq, uniform_initial = False):
        # print(self.model.startprob_prior)
        if uniform_initial:
            self.model.startprob_ = np.ones(self.model.transmat_.shape[0])/self.model.transmat_.shape[0]
        else:
            self.model.startprob_ = self.startprob_
        last_state_proba = self.model.predict_proba(obs_seq.reshape(-1,1))
        # print(self.model.emissionprob_.T.shape, self.model.transmat_.T.shape, last_state_proba.shape)
        return np.matmul(self.model.emissionprob_.T, np.matmul(self.model.transmat_.T, last_state_proba.T))