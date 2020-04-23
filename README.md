# CompExam_Artifact
Lenskit is a set of Python tools for training, running, and evaluating recommender systems. The Lenskit modules make it easier and more efficient to use recommender system algorithms. Lenskit provides libraries for implementing collaborative filtering algorithms. These algorithms generate recommendations based on explicit or implicit user ratings.

In this project, two content based recommender systems are implemented where item features are generated from review text. For a given user and a list of items, score is predicted based on item-item similarity.

## Requirements

* python
* pip
* numpy
* pandas
* nltk
* scikit-learn
* scipy
* matplotlib
* lenskit
* logging

## Installing Lenskit

To install the current release with Anaconda (recommended):
conda install -c lenskit lenskit
Or you can use pip:
pip install lenskit
## Loading Data and Sampling
The algorithms need a dataset that has user and item interactions. For this project, Steam game reviews dataset is used. After loading the data, users who have at least 5 items in their list are selected. 
Crossfold methods from LensKit are used to split the dataset into train and test sets.
## Generate Recommendations and Evaluation
After defining the algorithms to generate recommendations for users, lenskit.batch is used to run the experiment.
For evaluation, mean reciprocal rank (MRR) and recall are calculated for each algorithm. Comparison among the algorithms are shown in the bar chart.

