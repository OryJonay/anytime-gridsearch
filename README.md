# AnyTimeGridSearchCV

> An [anytime](https://en.wikipedia.org/wiki/Anytime_algorithm) implementation of scikit-learn [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).

[![Build Status](https://travis-ci.org/OryJonay/anytime-gridsearch.svg?branch=develop)](https://travis-ci.org/OryJonay/anytime-gridsearch) [![Coverage Status](https://coveralls.io/repos/github/OryJonay/anytime-gridsearch/badge.svg?branch=develop)](https://coveralls.io/github/OryJonay/anytime-gridsearch?branch=develop)

## Motivation

Waiting for GridSearchCV to finish running can be quite long, using an anytime approach will allow the algorithm to run in the background, with an endpoint to query for best result.

## Brief overview

The project consists of the following parts:

1. A web application for creating and displaying searches and results through a REST API
2. A distributed cluster for running the searches
3. A [SPA](https://en.wikipedia.org/wiki/Single-page_application) built with [Vue.js](https://vuejs.org/v2/guide/) for web gui


## Installation

The project requires:

* Python (>=3.5)
* Django (>=1.11)
* PostgreSQL (>=9.6)
* distributed (>=1.18.3)

``` bash
# clone repo
git clone https://github.com/OryJonay/anytime-gridsearch.git anytimegridsearch

# create virtual environment
cd anytimegridsearch
virtualenv -p python3.5 .

# install dependencies
pip install -r requirements.txt

# run tests
python manage.py test
```

## Usage
The old way of using scikit-learn's GridSearchCV (taken from the examples part of [the documentation](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)):
``` python
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
iris = datasets.load_iris()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(iris.data, iris.target)
print clf.best_params_['kernel']
```
This will run all possible grid points (4 in this example), and only after all grid points are fitted and cross validated will return (and print the best kernel).
We'll do it like this:
``` python
from sklearn import svm, datasets
from AnyTimeGridSearchCV.grids.anytime_search import AnyTimeGridSearchCV as GridSearchCV
iris = datasets.load_iris()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(iris.data, iris.target)
print clf.best_params_['kernel']
```
And that's (mostly) it- just change the search algorithm to the new one, and voilà- all done!
The call to the search algorithm is non blocking, so it's possible to query the search algorithm before all the grid points are cross validated.

## Roadmap

Things to do in the future (not sorted by priority):

* Docker images for project
* Finish web gui
* Web gui unit tests & end to end tests with backend
