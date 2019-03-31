import numpy as np
import pandas as pd
import collections
from scipy import stats


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
        self._E = self.data
        self.compute_selectors()

        # This list will contains the complex-class pairs that will represent the rules found by the CN2 algorithm.
        rule_list = []

        while len(self._E) > 0:
            best_cpx = self.find_best_complex()
            print('New best complex: ' + str(best_cpx))
            if best_cpx is not None:
                covered_examples = self.get_covered_examples(best_cpx)
                most_common_class = self.get_most_common_class(covered_examples)
                self.remove_examples(covered_examples)
                rule_list.append((best_cpx, most_common_class))
            else:
                if len(self._E) > 0:
                    rule_list.append((None, self.get_most_common_class(self._E)))
                break

        return rule_list

    def predict(self, csv_test, rule_list):
        # TODO: create function to predict the labels based on the previously created rule set
        return

    def compute_selectors(self):
        """
        This function computes the selectors from the input data, which are
        the pairs attribute-value, excluding the class attribute.
        Assumption: the class attribute is the last attribute of the dataset.
        """
        #print('$ compute_selectors')
        attributes = list(self.data)

        # removing the class attribute
        del attributes[-1]

        for attribute in attributes:
            possible_values = set(self.data[attribute])
            for value in possible_values:
                self._selectors.append((attribute, value))

    def find_best_complex(self):
        # TODO: create function to find the best complex
        ##print('$ find_best_complex')
        best_complex = None
        best_complex_entropy = float('inf')
        best_complex_significance = 0
        star = []

        while True:
            entropy_measures = {}
            new_star = self.specialize_star(star, self._selectors)
            ##print('New star: ' + str(new_star))
            for idx in range(len(new_star)):
                tested_complex = new_star[idx]
                significance = self.significance(tested_complex)
                #print('Significance of ' + str(tested_complex) + ' : ' + str(significance))
                entropy = self.entropy(tested_complex)
                #print('Entropy of ' + str(tested_complex) + ' : ' + str(entropy))
                entropy_measures[idx] = entropy
                if significance > self.min_significance and \
                        entropy < best_complex_entropy:
                    #print('New best complex found')
                    best_complex = tested_complex.copy()
                    best_complex_entropy = entropy
                    best_complex_significance = significance
            top_complexes = sorted(entropy_measures.items(), key=lambda x: x[1], reverse=True)[:self.star_max_size]
            star = [new_star[x[0]] for x in top_complexes]
            ##print('Star: ' + str(star))
            if len(star) == 0 or best_complex_significance < self.min_significance:
                break

        return best_complex

    def remove_examples(self, indexes):
        '''
        Removes from E the covered examples with the indexes received as parameter.
        :param indexes: list of index labels that identify the instances to remove from E.
        '''
        ##print('$ remove_examples')
        self._E = self._E.drop(indexes)

    def get_covered_examples(self, best_cpx):
        '''
        Returns the indexes of the examples from in E covered by the complex.
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
        examples = self._E
        covered_examples = examples[examples.isin(values).all(axis=1)]

        return covered_examples.index

    def get_most_common_class(self, covered_examples):
        '''
        Returns the most common class among the examples received as parameter. It assumes that the class is the last
        attribute of the examples.
        :param covered_examples: Pandas DataFrame containing the examples from which we want to find the most common
        class.
        :return: label of the most common class.
        '''
        ##print('$ get_most_common_class')
        classes = self.data.loc[covered_examples, [list(self.data)[-1]]]
        most_common_class = classes.iloc[:,0].value_counts().idxmax()
        return most_common_class

    def specialize_star(self, star, selectors):
        ##print('$ specialize_star')
        ##print('## Star: ' + str(star))
        new_star = []
        if len(star) > 0:
            for complex in star:
                for selector in selectors:
                    new_complex = complex.copy()
                    new_complex.append(selector)
                    #print('New temporary complex: ' + str(new_complex))

                    # Add the new complex only if they are valid
                    count = collections.Counter([x[0] for x in new_complex])
                    duplicate = False
                    for c in count.values():
                        if c > 1:
                            duplicate = True
                            break
                    # TODO: check the condition 'new_complex not in star'
                    if not duplicate:
                        #print('Valid')
                        new_star.append(new_complex)
                        #print('New temporary new_star: ' + str(new_star))
        else:
            for selector in selectors:
                new_star.append([selector])
        #print('New star: ' + str(new_star))
        return new_star

    def significance(self, tested_complex):
        covered_examples = self.get_covered_examples(tested_complex)
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
        covered_examples = self.get_covered_examples(tested_complex)
        classes = self.data.loc[covered_examples, [list(self.data)[-1]]]
        num_instances = len(classes)
        class_counts = classes.iloc[:,0].value_counts()
        class_probabilities = class_counts.divide(num_instances)
        log2_of_classprobs = np.log2(class_probabilities)
        plog2p = class_probabilities.multiply(log2_of_classprobs)
        entropy = plog2p.sum() * -1

        return entropy

    def print_rules(self, rules):
        ##print('$ print_rules')
        rule_string = ''
        for rule in rules:
            complex = rule[0]
            if complex is not None:
                complex_class = rule[1]
                for idx in range(len(complex)):
                    if idx == 0:
                        rule_string += 'If '
                    rule_string += str(complex[idx][0]) + '=' + str(complex[idx][1])
                    if idx < len(complex)-1:
                        rule_string += ' and '
                rule_string += ', then class=' + complex_class
            else:
                rule_string += 'class=' + complex_class
            print(rule_string)
            rule_string = ''
