from __future__ import print_function

import os
import glob
import subprocess
from time import time
from operator import itemgetter
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import warnings


def get_csv_data(filename):
    """Get the csv developers data
    Args
    ----
    filename -- filename with path.

    Returns
    -------
    df -- DataFrame with data.
    """

    if os.path.exists(filename):
        df = pd.read_csv(filename, index_col=0)
    return df

def get_csv_array_data(filenames):
    """Get the csv smells data
    Args
    ----
    filenames -- folder with path.

    Returns
    -------
    df -- DataFrame with data.
    """
    frame = pd.DataFrame()
    list = []
    for file in filenames:
        df = pd.read_csv(file, index_col=None, header=0)
        list.append(df)

    frame = pd.concat(list)
    return frame

def encode_target(df, target_column):
    """Add column to df with integers for the target.

    Args
    ----
    df -- pandas DataFrame.
    target_column -- column to map to int, producing
                     new Target column.

    Returns
    -------
    df_mod -- modified DataFrame.
    targets -- list of target names.
    """
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod["Target"] = df_mod[target_column].replace(map_to_int)

    return (df_mod, targets)


def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")


def report(grid_scores, n_top=3):
    """Report top n_top parameters settings, default n_top=3.

    Args
    ----
    grid_scores -- output from grid or random search
    n_top -- how many to report, of top models

    Returns
    -------
    top_params -- [dict] top parameter settings found in
                  search
    """
    top_scores = sorted(grid_scores,
                        key=itemgetter(1),
                        reverse=True)[:n_top]
    #for i, score in enumerate(top_scores):
        #print("Model with rank: {0}".format(i + 1))
        #print(("Mean validation score: "
        #       "{0:.3f} (std: {1:.3f})").format(
        #    score.mean_validation_score,
        #    np.std(score.cv_validation_scores)))
        #print("Parameters: {0}".format(score.parameters))
        #print("")

    return top_scores[0].parameters


def run_gridsearch(X, y, clf, param_grid, cv=5):
    """Run a grid search for best Decision Tree parameters.

    Args
    ----
    X -- features
    y -- targets (classes)
    cf -- scikit-learn Decision Tree
    param_grid -- [dict] parameter settings to test
    cv -- fold of cross-validation, default 5

    Returns
    -------
    top_params -- [dict] from report()
    """
    grid_search = GridSearchCV(clf,
                               param_grid=param_grid,
                               cv=cv)
    start = time()
    grid_search.fit(X, y)

    #print(("\nGridSearchCV took {:.2f} "
    #       "seconds for {:d} candidate "
    #       "parameter settings.").format(time() - start,
    #                                     len(grid_search.grid_scores_)))

    top_params = report(grid_search.grid_scores_, 3)
    return top_params


def get_code(tree, feature_names, target_names,
             spacer_base="    "):
    """Produce pseudo-code for decision tree.

    Args
    ----
    tree -- scikit-leant Decision Tree.
    feature_names -- list of feature names.
    target_names -- list of target (class) names.
    spacer_base -- used for spacing code (default: "    ").

    Notes
    -----
    based on http://stackoverflow.com/a/30104792.
    """
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value


    def recurse(left, right, threshold, features, node, depth):
        global s_tree
        spacer = spacer_base * depth
        if (threshold[node] != -2):
            s_tree +=(spacer + "if ( " + features[node] + " <= " +
                  str(threshold[node]) + " ) {")
            if left[node] != -1:
                    recurse (left, right, threshold, features,
                             left[node], depth+1)
            s_tree += (spacer + "}\n" + spacer +"else {")
            if right[node] != -1:
                    recurse (left, right, threshold, features,
                             right[node], depth+1)
            s_tree +=(spacer + "}")
        else:
            target = value[node]
            for i, v in zip(np.nonzero(target)[1],
                            target[np.nonzero(target)]):
                target_name = target_names[i]
                target_count = int(v)
                s_tree +=(spacer + "return " + str(target_name) +
                      " ( " + str(target_count) + " examples )")

    recurse(left, right, threshold, features, 0, 0)


col_names = ["Type", "Developer", "Smell", "Folds", "Criterion", "Max_Depth", "Max_Leaf_Nodes", "Min_Samples_Leaf",
             "Min_Samples_Split", "Tree", "Accuracy", "Precision", "Recall", "Fmeasure"]
df_results = pd.DataFrame(columns=col_names)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    for folder in glob.iglob('csv/*'):
        files = []
        for filename in glob.iglob(folder + '/*.csv'):
            files.append(filename)
        if len(files) > 0:
            print('files: ' + str(files))
            s_tree = ''
            df = get_csv_array_data(files)
            df, targets = encode_target(df, "Smell")
            features = list(df.columns[:16])
            y = df["Target"]
            X = df[features]

            # New
            # print("-- Grid Parameter Search via 5-fold CV")
            # set of parameters to test
            param_grid = {"criterion": ["gini", "entropy"],
                          "min_samples_split": [2, 10, 20],
                          "max_depth": [None, 2, 5, 10],
                          "min_samples_leaf": [1, 5, 10],
                          "max_leaf_nodes": [None, 5, 10, 20],
                          }

            dt = DecisionTreeClassifier(random_state=99)
            ts_gs = run_gridsearch(X, y, dt, param_grid, cv=5)

            # print("\n-- Best Parameters:")
            # for k, v in ts_gs.items():
            #    print("parameter: {:<20s} setting: {}".format(k, v))

            # Test the retuned best parameters
            dt_ts_gs = DecisionTreeClassifier(**ts_gs, random_state=99)
            scores_acc = cross_val_score(dt_ts_gs, X, y, cv=5, scoring='accuracy')
            scores_pre = cross_val_score(dt_ts_gs, X, y, cv=5, scoring='precision')
            scores_rec = cross_val_score(dt_ts_gs, X, y, cv=5, scoring='recall')
            scores_f1 = cross_val_score(dt_ts_gs, X, y, cv=5, scoring='f1')

            #print("mean: {:.3f} (std: {:.3f})".format(scores_acc.mean(),
            #                                          scores_acc.std()),
            #      end="\n\n")

            get_code(dt_ts_gs.fit(X, y), features, targets)
            l_results = ["Type", '', folder.split("/")[1], 5, ts_gs['criterion'],
                         ts_gs['max_depth'], ts_gs['max_leaf_nodes'], ts_gs['min_samples_leaf'],
                         ts_gs['min_samples_split'], s_tree, np.mean(scores_acc), np.mean(scores_pre), np.mean(scores_rec),
                         np.mean(scores_f1)]
            df_results = df_results.append(pd.Series(l_results, index=col_names), ignore_index=True)

#Output Smells
df_results.to_csv("smells.csv", encoding='utf-8')
print("smells.csv generated.... \n")

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    for folder in glob.iglob('csv/*'):
        for filename in glob.iglob(folder + '/*.csv'):
            print("file: " + filename)
            s_tree = ''
            df = get_csv_data(filename)
            df, targets = encode_target(df, "Smell")
            features = list(df.columns[:16])
            y = df["Target"]
            X = df[features]

            # New
            #print("-- Grid Parameter Search via 5-fold CV")
            # set of parameters to test
            param_grid = {"criterion": ["gini", "entropy"],
                          "min_samples_split": [2, 10, 20],
                          "max_depth": [None, 2, 5, 10],
                          "min_samples_leaf": [1, 5, 10],
                          "max_leaf_nodes": [None, 5, 10, 20],
                          }

            dt = DecisionTreeClassifier(random_state=99)
            ts_gs = run_gridsearch(X, y, dt, param_grid, cv=5)

            #print("\n-- Best Parameters:")
            #for k, v in ts_gs.items():
            #    print("parameter: {:<20s} setting: {}".format(k, v))

            # Test the retuned best parameters
            dt_ts_gs = DecisionTreeClassifier(**ts_gs,random_state=99)
            scores_acc = cross_val_score(dt_ts_gs, X, y, cv=5, scoring='accuracy')
            scores_pre = cross_val_score(dt_ts_gs, X, y, cv=5, scoring='precision')
            scores_rec = cross_val_score(dt_ts_gs, X, y, cv=5, scoring='recall')
            scores_f1 = cross_val_score(dt_ts_gs, X, y, cv=5, scoring='f1')

            #print("mean: {:.3f} (std: {:.3f})".format(scores_acc.mean(),
            #                                          scores_acc.std()),
            #                                                end="\n\n")

            get_code(dt_ts_gs.fit(X,y), features, targets)
            l_results = ["Type", filename.split(" - ")[1].split(".csv")[0], filename.split("/")[1], 5, ts_gs['criterion'], ts_gs['max_depth'], ts_gs['max_leaf_nodes'], ts_gs['min_samples_leaf'],
                         ts_gs['min_samples_split'], s_tree, np.mean(scores_acc), np.mean(scores_pre), np.mean(scores_rec), np.mean(scores_f1)]
            df_results = df_results.append(pd.Series(l_results, index=col_names), ignore_index=True)
#Output Devs
df_results.to_csv("developers.csv", encoding='utf-8')
df_results = pd.DataFrame(columns=col_names)
print("developers.csv generated.... \n")