# CN2 rule-based classifier
Author: Alessia Mondolo


This repository contains my implementation for the CN2 rule-based classifer.

The folders of this project are structured as follows:
- 'Documentation/': this folder contains the file 'report.pdf', which describes:
	- the algorithm of the rule-based classifier;
	- the evaluation of the results obtained for the tested datasets;
	- how to execute the code;
	- other comments and conclusions on this project.
- 'Data/': this colder contains the original datasets used for training and testing the rule-based classifier:
	- 'arff': this folder contains the original datasets in .arff format
	- 'csv': this folder contains the preprocessed datasets, saved in .csv format
	- 'output': this folder contains the output files
- 'Source/': this folder contains the source code of the project:
	- 'preprocessing.py': this file contains the code for the preprocessing of the data sets;
	- 'CN2.py': this file contains the class for the CN2 rule-based classifier;
	- 'tests.py': this file contains the tests run with the classifier for all the data sets.
