from itertools import combinations
import json
import matplotlib.pyplot as plt
import numpy as np
from ordered_set import OrderedSet
import pandas as pd
import pickle
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import CategoricalNB, GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier

# def string_to_int(df, column_name):
#     categories = list(OrderedSet(df[column_name]))
#     categories.sort(key=lambda category: df[column_name].value_counts()[category], reverse=True)
#     encoding = {category: categories.index(category) for category in categories}
#     return encoding


def naive_bayes(X_train, y_train, X_test):
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    return y_pred


def decision_tree(X_train, y_train, X_test):
    tree = DecisionTreeClassifier()
    y_pred = tree.fit(X_train, y_train).predict(X_test)
    return y_pred

# def neural_network(X_train, y_train, X_test):
#     nnw = MLPClassifier(max_iter=500)
#     y_pred = nnw.fit(X_train, y_train).predict(X_test)
#     return y_pred


def performance(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    return acc, prec, recall, f1


def get_feat_combinations(features):
    combos = [[] for _ in features]
    for i, r in enumerate(range(1, len(features)+1)):
        combos[i] = (list(combinations(features, r)))
    return [list(combo) for combo_list in combos for combo in combo_list]


# def find_best_categories(clf_function, df, target, feat_combinations):
#     perf_metrics = np.zeros((len(feat_combinations), 4), dtype=float)
#     print(len(feat_combinations))
#     for i, combination in enumerate(feat_combinations):
#         X = df[combination]
#         y = df[target]
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
#         y_pred = clf_function(X_train, y_train, X_test)
#         acc, prec, recall, f1 = performance(y_test, y_pred)
#         perf_metrics[i] = [acc, prec, recall, f1]
#         # print(acc, '\t', prec, '\t', recall, '\t', f1)
#     return perf_metrics


# def optimize_clf(clf_function, df, target, feat_combinations):

#     perf_metrics = np.zeros((len(feat_combinations), 4), dtype=float)
#     print(len(feat_combinations))
#     for i, combination in enumerate(feat_combinations):
#         X = df[combination]
#         y = df[target]
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
#         y_pred = clf_function(X_train, y_train, X_test)
#         acc, prec, recall, f1 = performance(y_test, y_pred)
#         perf_metrics[i] = [acc, prec, recall, f1]
#         # print(acc, '\t', prec, '\t', recall, '\t', f1)
#     return perf_metrics


def grid_search(clf, df, target, param_grid, feat_combinations):
    precision_scorer = make_scorer(precision_score, average='weighted', zero_division=0)
    best_score = 0
    best_gr_search = None
    for combination in feat_combinations:
        X = df[combination]
        y = df[target]
        gr_search = GridSearchCV(clf, param_grid, scoring=precision_scorer)
        gr_search.feats = combination
        gr_search.fit(X, y)
        if gr_search.best_score_ > best_score:
            best_score = gr_search.best_score_
            best_gr_search = gr_search

    class_name = str(clf.__class__).strip('<>').replace('class', '').strip('\' ').split('.')[-1]
    with open(f'best_{class_name}.pkl', 'wb') as augurk:
        pickle.dump(best_gr_search, augurk)

    return best_gr_search.best_score_, best_gr_search.best_params_, gr_search.feats
 

if __name__ == "__main__":

    # Load lengthy data from json file
    with open("supplementary_data.json", "r") as info:
        suppl_data = json.load(info)

    questions = suppl_data['questions']
    df = pd.read_excel('cleaned_dataset.xlsx')

    quantile_discretization = ["age", "What makes a good day for you (1)?", "hours of sleep", 
                               "How many hours per week do you do sports (in whole hours)? ", 
                               "How many students do you estimate there are in the room?", "Give a random number"]
    
    quant_disc = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
    df[quantile_discretization] = quant_disc.fit_transform(df[quantile_discretization])
    # print(quant_disc.bin_edges_)
    # raise NotImplementedError
    uniform_disc = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform')
    df[questions["stress_level"]] = uniform_disc.fit_transform(df[questions["stress_level"]].values.reshape(-1, 1))


    feat_enc = OrdinalEncoder()
    # label_enc = LabelEncoder()
    df[[questions["master_program"], questions["gender"]]] = feat_enc.fit_transform(df[[questions["master_program"], questions["gender"]]])
    # print(df[questions["master_program"]])
    target = questions["stress_level"]
    # print(set([target]))
    # print(OrderedSet(df.columns))
    features = OrderedSet(df.columns).difference(OrderedSet([target]))
    # print(list(features))
    # raise Exception
    # print(features, len(features))
    # raise NotImplementedError
    
    # print(feat_combinations)
    # perf_scores = find_best_categories(naive_bayes, df, target, feat_combinations)
    # perf_scores = find_best_categories(neural_network, df, target, feat_combinations)
    # print(df)

    # print(perf_scores.shape[0])
    # print(search)
    # perf_metrics = ["acc", "prec", "recall", "f1"]
    # best_combo_indices = {}

    '''
    for i in range(4):
        arg_array = np.argsort(perf_scores[:,i])[::-1]
        best_indices = arg_array[:10]
        min_arg = np.min(best_indices)
        print(perf_scores[best_indices])
        best_combo_indices[perf_metrics[i]] = best_indices, min_arg
    winning_indices = best_combo_indices.values()
    print(winning_indices)
    '''

    # X = df[features]
    # y = df[target]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # y_pred = decision_tree(X_train, y_train, X_test)
    # acc, prec, recall, f1 = performance(y_test, y_pred)
    # print(acc, '\n', prec, '\n', recall, '\n', f1)
    # y_pred = naive_bayes(X_train, y_train, X_test)
    # acc, prec, recall, f1 = performance(y_test, y_pred)
    # print(acc, '\n', prec, '\n', recall, '\n', f1)

    # feat_combinations = get_feat_combinations(features)
    # for clf_param in [[GaussianNB(), {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]}],
    #                   [DecisionTreeClassifier(), {'max_depth': [5, 10, 15, 20], 'min_samples_split': [5, 10, 15, 20], 
    #                                               'min_samples_leaf': [1, 2, 4]}]]: 
    #     clf, param_grid = clf_param
    #     search = grid_search(clf, df.copy(deep=True), target, param_grid, feat_combinations.copy())
    #     print(search)

    clfs = [GaussianNB(), DecisionTreeClassifier()]

    for clf in clfs:
        class_name = str(clf.__class__).strip('<>').replace('class', '').strip('\' ').split('.')[-1]
        with open(f'best_{class_name}.pkl', 'rb') as result:
            gr_search = pickle.load(result)
            print(gr_search.feats)
            print(gr_search.best_params_)
            print(gr_search.best_score_)

