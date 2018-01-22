# AnyTimeGridSearchCV

> An [anytime](https://en.wikipedia.org/wiki/Anytime_algorithm) implementation of scikit-learn [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).

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

## Roadmap

Things to do in the future (not sorted by priority):

* Docker images for project
* Finish web gui
* Web gui unit tests & end to end tests with backend
