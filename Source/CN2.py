import numpy as np
import pandas as pd
import collections
import time
import pickle
from sklearn.metrics import accuracy_score


# TODO: check copy of list insted of passing by reference
class CN2:

    _dataPath = '../Data/csv/'
    _E = []
    _selectors = []

    def __init__(self, star_max_size=5, min_significance = 0.5):
        self.data = None
        self.star_max_size = star_max_size
        self.min_significance = min_significance

    def fit(self, file_name):
        """
        This function is used to learn the rule-based classification model with the CN2 algorithm.
        :param file_name: the name of the training file in CSV format.
        The file must be located in the '../Data/csv/' folder.
        """
        self.data = pd.read_csv(self._dataPath + file_name)
        self._E = self.data.copy()
        self.compute_selectors()

        # This list will contains the complex-class pairs that will represent the rules found by the CN2 algorithm.
        rule_list = []
        classes = self.data.loc[:, [list(self.data)[-1]]]
        classes_count = classes.iloc[:,0].value_counts()

        while len(self._E) > 0:
            best_cpx = self.find_best_complex()
            if best_cpx is not None:
                covered_examples = self.get_covered_examples(self._E, best_cpx)
                coverage = len(covered_examples)
                most_common_class, count = self.get_most_common_class(covered_examples)
                self._E = self.remove_examples(self._E, covered_examples)

                # Precision: how many covered examples belong to the most common class
                total = 0
                if most_common_class in classes_count.keys():
                    total = classes_count[most_common_class]
                precision = count / total

                rule_list.append((best_cpx, most_common_class, coverage, precision))
            else:
                print('######## Best complex is None! ########')
                if len(self._E) > 0:
                    most_common_class, count = self.get_most_common_class(self._E.index)
                    total = 0
                    if most_common_class in classes_count.keys():
                        total = classes_count[most_common_class]
                    precision = count / total
                    rule_list.append((None, most_common_class, len(self._E), precision))
                break

        most_common_class, count = self.get_most_common_class(self.data.index)
        total = classes_count[most_common_class]
        precision = count / total
        rule_list.append((None, most_common_class, count, precision))

        return rule_list

    def predict(self, test_file_name, rule_list):
        # TODO: create function to predict the labels based on the previously created rule set
        test_data = pd.read_csv(self._dataPath + test_file_name)
        test_classes = test_data.iloc[:, -1].values
        test_data = test_data.iloc[:, :-1]
        predicted_classes = [None] * len(test_classes)
        rules_performance = []
        remaining_examples = test_data.copy()

        for rule in rule_list:
            rule_complex = rule[0]
            if rule_complex is not None:
                covered_examples = self.get_covered_examples(remaining_examples, rule_complex)
                remaining_examples = self.remove_examples(remaining_examples, covered_examples)
                indexes = list(covered_examples)
                predicted_class = rule[1]
                correct_predictions = 0
                wrong_predictions = 0
                for index in indexes:
                    predicted_classes[index] = predicted_class
                    if test_classes[index] == predicted_class:
                        correct_predictions += 1
                    else:
                        wrong_predictions += 1
                sums = correct_predictions + wrong_predictions
                if sums > 0:
                    accuracy = str(correct_predictions / sums)
                else:
                    accuracy = '-'
                performance = {'rule': rule,
                               'predicted class': predicted_class,
                               'covered examples': len(indexes),
                               'correct predictions': correct_predictions,
                               'wrong predictions': wrong_predictions,
                               'rule accuracy': accuracy}
                rules_performance.append(performance)
            else:
                if len(remaining_examples) > 0:
                    print('Using default class')
                    indexes = list(remaining_examples.index)
                    predicted_class = rule[1]
                    correct_predictions = 0
                    wrong_predictions = 0
                    for index in indexes:
                        predicted_classes[index] = predicted_class
                        if test_classes[index] == predicted_class:
                            correct_predictions += 1
                        else:
                            wrong_predictions += 1
                    sums = correct_predictions + wrong_predictions
                    if sums > 0:
                        accuracy = str(correct_predictions / sums)
                    else:
                        accuracy = '-'
                    performance = {'rule': rule,
                                   'predicted class': predicted_class,
                                   'covered examples': len(indexes),
                                   'correct predictions': correct_predictions,
                                   'wrong predictions': wrong_predictions,
                                   'rule accuracy': accuracy}
                    rules_performance.append(performance)

        return rules_performance, accuracy_score(test_classes, predicted_classes)

    def compute_selectors(self):
        """
        This function computes the selectors from the input data, which are
        the pairs attribute-value, excluding the class attribute.
        Assumption: the class attribute is the last attribute of the dataset.
        """
        attributes = list(self.data)

        # removing the class attribute
        del attributes[-1]

        for attribute in attributes:
            possible_values = set(self.data[attribute])
            for value in possible_values:
                self._selectors.append((attribute, value))

    def find_best_complex(self):
        best_complex = None
        best_complex_entropy = float('inf')
        best_complex_significance = 0
        star = []

        while True:
            entropy_measures = {}
            new_star = self.specialize_star(star, self._selectors)
            for idx in range(len(new_star)):
                tested_complex = new_star[idx]
                significance = self.significance(tested_complex)
                entropy = self.entropy(tested_complex)
                entropy_measures[idx] = entropy
                if significance > self.min_significance and \
                        entropy < best_complex_entropy:
                    best_complex = tested_complex.copy()
                    best_complex_entropy = entropy
                    best_complex_significance = significance
            top_complexes = sorted(entropy_measures.items(), key=lambda x: x[1], reverse=False)[:self.star_max_size]
            star = [new_star[x[0]] for x in top_complexes]
            if len(star) == 0 or best_complex_significance < self.min_significance:
                break

        return best_complex

    def remove_examples(self, all_examples, indexes):
        '''
        Removes from E the covered examples with the indexes received as parameter.
        :param indexes: list of index labels that identify the instances to remove from E.
        '''
        remaining_examples = all_examples.drop(indexes)
        return remaining_examples

    def get_covered_examples(self, all_examples, best_cpx):
        '''
        Returns the indexes of the examples from in E covered by the complex.
        :param all_examples:
        :param best_cpx: list of attribute-value tuples.
        :return:
        '''
        # Creating a dictionary with the attributes of the best complex as key, and the values of that attribute as a
        # list of values. Then, add all the possible values for the attributes that are not part of the rules of the
        # best complex.
        values = dict()
        [values[t[0]].append(t[1]) if t[0] in list(values.keys())
         else values.update({t[0]: [t[1]]}) for t in best_cpx]
        for attribute in list(self.data):
            if attribute not in values:
                values[attribute] = set(self.data[attribute])

        # Getting the indexes of the covered examples
        covered_examples = all_examples[all_examples.isin(values).all(axis=1)]
        return covered_examples.index

    def get_most_common_class(self, covered_examples):
        '''
        Returns the most common class among the examples received as parameter. It assumes that the class is the last
        attribute of the examples.
        :param covered_examples: Pandas DataFrame containing the examples from which we want to find the most common
        class.
        :return: label of the most common class.
        '''
        classes = self.data.loc[covered_examples, [list(self.data)[-1]]]
        most_common_class = classes.iloc[:,0].value_counts().index[0]
        count = classes.iloc[:,0].value_counts()[0]
        return most_common_class, count

    def specialize_star(self, star, selectors):
        new_star = []
        if len(star) > 0:
            for complex in star:
                for selector in selectors:
                    new_complex = complex.copy()
                    new_complex.append(selector)

                    # Add the new complex only if they are valid
                    count = collections.Counter([x[0] for x in new_complex])
                    duplicate = False
                    for c in count.values():
                        if c > 1:
                            duplicate = True
                            break
                    # TODO: check the condition 'new_complex not in star'
                    if not duplicate:
                        new_star.append(new_complex)
        else:
            for selector in selectors:
                new_star.append([selector])
        return new_star

    def significance(self, tested_complex):
        covered_examples = self.get_covered_examples(self._E, tested_complex)
        classes = self.data.loc[covered_examples, [list(self.data)[-1]]]
        covered_num_instances = len(classes)
        covered_counts = classes.iloc[:,0].value_counts()
        covered_probs = covered_counts.divide(covered_num_instances)

        train_classes = self.data.iloc[:,-1]
        train_num_instances = len(train_classes)
        train_counts = train_classes.value_counts()
        train_probs = train_counts.divide(train_num_instances)

        significance = covered_probs.multiply(np.log(covered_probs.divide(train_probs))).sum() * 2

        return significance

    def entropy(self, tested_complex):
        covered_examples = self.get_covered_examples(self._E, tested_complex)
        classes = self.data.loc[covered_examples, [list(self.data)[-1]]]
        num_instances = len(classes)
        class_counts = classes.iloc[:,0].value_counts()
        class_probabilities = class_counts.divide(num_instances)
        log2_of_classprobs = np.log2(class_probabilities)
        plog2p = class_probabilities.multiply(log2_of_classprobs)
        entropy = plog2p.sum() * -1

        return entropy

    def print_rules(self, rules):
        rule_string = ''
        for rule in rules:
            complex = rule[0]
            complex_class = rule[1]
            coverage = rule[2]
            precision = rule[3]

            if complex is not None:
                for idx in range(len(complex)):
                    if idx == 0:
                        rule_string += 'If '
                    rule_string += str(complex[idx][0]) + '=' + str(complex[idx][1])
                    if idx < len(complex)-1:
                        rule_string += ' and '
                rule_string += ', then class=' + complex_class + ' [covered examples = ' + str(coverage) + ', precision = ' \
                               + str(precision) + ']'
            else:
                rule_string += 'Default: class=' + complex_class
            print(rule_string)
            rule_string = ''


if __name__ == "__main__":
    cn2 = CN2()
    train_start = time.time()
    rules = cn2.fit('zoo_train.csv')
    train_end = time.time()
    print('Training time: ' + str(train_end-train_start) + ' s')
    print('Rules:')
    cn2.print_rules(rules)

    with open('../Data/output/rules', 'wb') as f:
        pickle.dump(rules, f)

    rules_performance, accuracy = cn2.predict('zoo_test.csv', rules)
    print('Accuracy: ', accuracy)
    print('Testing performance:')
    keys = []
    vals = []
    for data in rules_performance:
        val = []
        for k, v in data.items():
            keys.append(k)
            val.append(v)
        vals.append(val)

    table = pd.DataFrame([v for v in vals], columns=list(dict.fromkeys(keys)))
    print(table)
    table.to_csv('../Data/output/zoo_performance.csv')
