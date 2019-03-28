import pandas as pd
from scipy import stats


class CN2:

    _dataPath = '../Data/csv/'
    _E = []
    _selectors = []

    def __init__(self, k=5):
        self.data = None
        self.k = k

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
            if best_cpx is not None:
                covered_examples = self.get_covered_examples(best_cpx)
                most_common_class = self.get_most_common_class(covered_examples)
                self.remove_examples(covered_examples)
                rule_list.append((best_cpx, most_common_class))
            else:
                break

        return rule_list

    def predict(self, csv_test, rule_list):
        return

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
        new_E = []
        return new_E

    def remove_examples(self, covered_examples):
        return

    def get_covered_examples(self, best_cpx):
        return 'test'

    def get_most_common_class(self, covered_examples):
        most_common_class = 'most common class'
        return most_common_class

    def print_rules(self, rules):
        print('Printing the rules in a fancy format')
        return
