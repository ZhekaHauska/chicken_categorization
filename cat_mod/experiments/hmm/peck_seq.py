import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cat_mod.models.hmm import HMMFitter


def prepare_data(path, drop_cols, drop_rows, sheet):
    df = pd.read_excel('data/seuqeunces_final_11_06_25.xlsx', sheet_name=sheet)
    df = df.drop(drop_cols, axis=1).drop(drop_rows, axis=0)
    return df.iloc[:, :80], df.iloc[:, 80:]

def map_colours(data, nan=-1, min_count=0):
    colours = data.stack().value_counts()
    col_to_numb = dict()
    for i, col in enumerate(colours.index):
        if colours.loc[col] > min_count:
            col_to_numb[col] = i
        else:
            col_to_numb[col] = nan
    data.replace(col_to_numb, inplace=True)
    data.fillna(nan, inplace=True)
    return data, col_to_numb, colours


def main():
    no_red1_train, no_red1_test = prepare_data('data/seuqeunces_final_11_06_25.xlsx', ['name'], [0, 1, 3, 4], sheet=0)
    no_red2_train, no_red2_test = prepare_data('data/seuqeunces_final_11_06_25.xlsx', ['name'], [0, 5], sheet=2)
    no_red_train = pd.concat([no_red1_train, no_red2_train]).reset_index(drop=True)
    no_red_train_num, color_to_num, freqs = map_colours(no_red_train, min_count=3, nan=0)
    print(color_to_num)
    print(freqs)
    X = no_red_train.to_numpy().astype(np.int32)

    conf = dict(
        n_components=[2, 5, 10, 20, 40],
        seeds = [32, 432, 333],
        # n_components = [2],
        # seeds = [42],
        max_iter=1000,
        tol=0.1,
        filter_colors = [0],
        account_frequencies=False,
        max_cells_per_column=20,
        min_cells_per_column=2,
        verbose=False
    )

    model_wrapper = HMMFitter(X)
    model_wrapper.fit_model(
        **conf
    )

    plt.errorbar(
        pd.unique(model_wrapper.scores['n_comp']),
        model_wrapper.scores.groupby(model_wrapper.scores['n_comp']).mean().to_numpy().flatten(),
        yerr=model_wrapper.scores.groupby(model_wrapper.scores['n_comp']).std().to_numpy().flatten(),
        label='unshuffled'
    )

    rng = np.random.default_rng()
    X_shuffled = rng.permuted(X, axis=1)

    shuffled_wrapper = HMMFitter(X_shuffled)
    shuffled_wrapper.fit_model(
        **conf
    )

    plt.errorbar(
        pd.unique(shuffled_wrapper.scores['n_comp']),
        shuffled_wrapper.scores.groupby(shuffled_wrapper.scores['n_comp']).mean().to_numpy().flatten(),
        yerr=shuffled_wrapper.scores.groupby(shuffled_wrapper.scores['n_comp']).std().to_numpy().flatten(),
        label='shuffled')

    plt.xticks(pd.unique(model_wrapper.scores['n_comp']))
    plt.xlabel('n components')
    plt.ylabel('score')
    plt.xscale('log')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()