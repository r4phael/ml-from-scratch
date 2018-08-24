import csv
import glob
import os
import pandas as pd
from decision_tree import generate_confusion_matrix
from decision_tree import array_generate_confusion_matrix

#os.remove('results.txt')


def println(text):
    print(text)
    file = open('results.txt', 'a')
    file.write(text)
    file.write('\n')
    file.close()


# Load a CSV file
def g_load_csv(filename):
    with open(filename, "rt") as f:
        reader = csv.reader(f)
        headers = next(reader, None)
        #headers = reader.next()
        dataset = [row for row in reader]
        return {'headers': headers, 'dataset': dataset}
        # lines = reader(file)
        # dataset = list(lines)


# Load a CSV file
def g_load_csv_array(files):
    dataset = []
    headers = []
    for file in files:
        with open(file, "rt") as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            #headers = reader.next()
            dataset += [row for row in reader]
    return {'headers': headers, 'dataset': dataset}


# Convert string column to float
def g_str_column_to_float(dataset, column):
    for row in dataset:
        if row[column].strip() == 'FALSE':
            row[column] = float(0)
        elif row[column].strip() == 'TRUE':
            row[column] = float(1)
        else:
            row[column] = float(row[column].strip())


# g_split a dataset based on an attribute and an attribute value
def g_test_g_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# Calculate the Gini index for a g_split dataset
def g_gini_index(groups, classes):
    # count all samples at g_split point
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


# Select the best g_split point for a dataset
def g_get_g_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = g_test_g_split(index, row[index], dataset)
            gini = g_gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


# Create a terminal node value
def g_to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


# Create child g_splits for a node or make terminal
def g_split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del (node['groups'])
    # check for a no g_split
    if not left or not right:
        node['left'] = node['right'] = g_to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = g_to_terminal(left), g_to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = g_to_terminal(left)
    else:
        node['left'] = g_get_g_split(left)
        g_split(node['left'], max_depth, min_size, depth + 1)
    # process right child
    if len(right) <= min_size:
        node['right'] = g_to_terminal(right)
    else:
        node['right'] = g_get_g_split(right)
        g_split(node['right'], max_depth, min_size, depth + 1)


# Build a decision tree
def g_build_tree(train, max_depth, min_size):
    root = g_get_g_split(train)
    g_split(root, max_depth, min_size, 1)
    return root


''' Old
# Print a decision tree
def g_print_tree(node, headers, depth=0):
    if isinstance(node, dict):
        println('%s[X%s < %.3f]' % (depth * ' ', (headers[node['index'] + 1]), node['value']))
        g_print_tree(node['left'], headers, depth + 1)
        g_print_tree(node['right'], headers, depth + 1)
    else:
        println('%s[%s]' % (depth * ' ', node))
'''

# Print a decision tree
def g_print_tree(node, headers, depth=0):
    global s_tree
    if isinstance(node, dict):
        s_tree += (depth * ' ' + '[' +(headers[node['index'] + 1]) + ' < ' + str(node['value']) + ']' + '\n')
        g_print_tree(node['left'], headers, depth + 1)
        g_print_tree(node['right'], headers, depth + 1)
    else:
        s_tree += (depth * ' ' + '[' + str(node) + '] \n')

def print_by_developer(filename, max_depth, min_size):

    #println("*********************************************************")
    #println("BY DEVELOPER CSV: " + filename)
    complete_dataset = g_load_csv(filename)
    dataset = complete_dataset['dataset']
    headers = complete_dataset['headers']

    for i in range(len(dataset[0])):
        g_str_column_to_float(dataset, i)

    tree = g_build_tree(dataset, max_depth, min_size)
    g_print_tree(tree, headers)
    l_results.append(s_tree)


def print_by_smell(files, max_depth, min_size):
    #println("*********************************************************")
    #println("BY SMELL CSV GROUP: " + ' '.join(files))
    complete_dataset = g_load_csv_array(files)
    dataset = complete_dataset['dataset']
    headers = complete_dataset['headers']
    for i in range(len(dataset[0])):
        g_str_column_to_float(dataset, i)

    tree = g_build_tree(dataset, max_depth, min_size)
    g_print_tree(tree, headers)
    l_results.append(s_tree)



max_depth = 5
min_size = 5
n_folds = 10
s_tree = ''
col_names = ["Type", "Filename", "Folds", "Max_Depth", "Min_Size", "Tree","TP", "FP", "FN", "Accuracy", "Precision", "Recall", "Fmeasure"]
df = pd.DataFrame(columns=col_names)

for folder in glob.iglob('csv/*'):
    for filename in glob.iglob(folder + '/*.csv'):
        l_results = list()
        l_results.extend(("Developer", filename, n_folds, max_depth, min_size))
        print_by_developer(filename, max_depth, min_size)
        scores = generate_confusion_matrix(filename, max_depth, min_size, n_folds)
        l_results.extend(scores.values())
        df = df.append(pd.Series(l_results, index=col_names), ignore_index=True)

#Output
df.to_csv("developers.csv", encoding='utf-8')
df = pd.DataFrame(columns=col_names)
println("developers.csv generated....")

for folder in glob.iglob('csv/*'):
    files = []
    for filename in glob.iglob(folder + '/*.csv'):
        files.append(filename)
    if len(files) > 0:
        l_results = list()
        l_results.extend(("Smells", files, n_folds, max_depth, min_size))
        print_by_smell(files,max_depth, min_size)
        scores = array_generate_confusion_matrix(files, max_depth, min_size, n_folds)
        l_results.extend(scores.values())
        df = df.append(pd.Series(l_results, index=col_names), ignore_index=True)

#Output
df.to_csv("smells.csv", encoding='utf-8')
println("smells.csv generated....")