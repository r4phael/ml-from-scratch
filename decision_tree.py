# F-MEASURE
# CART on the Bank Note dataset
from random import seed
from random import randrange
from collections import OrderedDict
import csv


def println(text):
    print(text)
    file = open('results.txt', 'a')
    file.write(text)
    file.write('\n')
    file.close()


# Load a CSV file
def load_csv(filename):
    with open(filename, "rt") as f:
        reader = csv.reader(f)
        headers = next(reader, None)
        dataset = [row for row in reader]
        return {'headers': headers, 'dataset': dataset}
        # lines = reader(file)
        # dataset = list(lines)


# Load a CSV file
def load_csv_array(files):
    dataset = []
    headers = []
    for file in files:
        with open(file, "rt") as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            dataset += [row for row in reader]
    return {'headers': headers, 'dataset': dataset}


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        if row[column].strip() == 'TRUE':
            row[column] = float(1)
        elif row[column].strip() == 'FALSE':
            row[column] = float(0)
        else:
            row[column] = float(row[column].strip())


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

#calculate precision:
def confusion_matrix(actual, predicted, type) :
   tp = 0.0
   fp = 0.0
   fn = 0.0
   tn = 0.0
   for i in range(len(actual)):
       if actual[i] == 1.0 and predicted[i] == 1.0:
           tp += 1
       elif actual[i] == 1.0 and predicted[i] == 0.0:
           fn += 1
       elif actual[i] == 0.0 and predicted[i] == 1.0:
           fp += 1
       elif actual[i] == 0.0 and predicted[i] == 0.0:
           tn += 1
   if (type == 1):
       return tp
   elif (type == 2):
       return fp
   else:
       return fn



# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, max_depth, min_size, headers):
    folds = cross_validation_split(dataset, n_folds)
    scores = []
    tp = fp = fn = 0.0
    mydict = OrderedDict()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, max_depth, min_size)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
        tp += confusion_matrix(actual, predicted, 1)
        fp += confusion_matrix(actual, predicted, 2)
        fn += confusion_matrix(actual, predicted, 3)
    mydict["tp"] = tp
    mydict["fp"] = fp
    mydict["fn"] = fn
    mydict["accuracy"] = sum(scores) / len(scores)
    mydict["precision"] = tp / (tp + fp) if (tp + fn) > 0 else 0
    mydict["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0
    mydict["f-measure"] = ((2 * tp) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0
    return mydict


# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini


# Select the best split point for a dataset
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


# Create a terminal node value
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del (node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth + 1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth + 1)


# Build a decision tree
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root


# Make a prediction with a decision tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


# Classification and Regression Tree Algorithm
def decision_tree(train, test, max_depth, min_size):
    tree = build_tree(train, max_depth, min_size)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return (predictions)


def generate_confusion_matrix(filename, max_depth, min_size, n_folds):
    # Test CART on Bank Note dataset
    seed(1)
    # load and prepare data
    complete_dataset = load_csv(filename)
    dataset = complete_dataset['dataset']
    headers = complete_dataset['headers']
    # convert string attributes to integers
    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)
    # evaluate algorithm
    scores = evaluate_algorithm(dataset, decision_tree, n_folds, max_depth, min_size, headers)
    return scores
    #println('Scores: %s' % scores)
    #print("Accuracy: ", scores['accuracy'], 'Precision: ', scores['precision'], 'Recall: ', scores['recall'], 'Fmeasure: ', scores['fmeasure'])


def array_generate_confusion_matrix(files, max_depth, min_size, n_folds):
    # Test CART on Bank Note dataset
    seed(1)
    # load and prepare data
    complete_dataset = load_csv_array(files)
    dataset = complete_dataset['dataset']
    headers = complete_dataset['headers']
    # convert string attributes to integers
    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)
    # evaluate algorithm
    scores = evaluate_algorithm(dataset, decision_tree, n_folds, max_depth, min_size, headers)
    return scores
    #print("Accuracy: ", scores['accuracy'], 'Precision: ', scores['precision'], 'Recall: ', scores['recall'], 'Fmeasure: ', scores['fmeasure'])

