import csv
import CN2

cn2 = CN2.CN2()
rules = cn2.fit('zoo.csv')

cn2.print_rules(rules)
