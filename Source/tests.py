import csv
import CN2
import time
import pickle
import pandas as pd

# cn2 = CN2.CN2()
# train_start = time.time()
# rules = cn2.fit('zoo_t1.csv')
# train_end = time.time()
# print('Training time: ' + str(train_end-train_start) + ' s')
# print('Rules:')
# cn2.print_rules(rules)
#
# test_start = time.time()
# rules_performance = cn2.predict('zoo_t1.csv', rules)
# test_end = time.time()
# print('Testing time: ' + str(test_end-test_start) + ' s')
# print('Testing performance:')
# print(rules_performance)
#
# with open('rules', 'wb') as f:
#     pickle.dump(rules, f)


with open('rules', 'rb') as f:
    rules = pickle.load(f)

cn2 = CN2.CN2()
cn2.data = pd.read_csv('../Data/csv/zoo.csv')
rules_performance, accuracy = cn2.predict('zoo_t1.csv', rules)
print('Accuracy: ', accuracy)
print('Testing performance:')
keys = []
vals = []
for data in rules_performance:
    val = []
    for k,v in data.items():
        keys.append(k)
        val.append(v)
    vals.append(val)

table = pd.DataFrame([v for v in vals], columns=list(dict.fromkeys(keys)))
print(table)
table.to_csv('../Data/csv/zoo_performance.csv')
