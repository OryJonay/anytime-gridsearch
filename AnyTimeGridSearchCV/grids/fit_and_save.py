'''
Created on Aug 10, 2017

@author: Ory Jonay
'''
import json
import os

import numpy
import requests
from sklearn.model_selection._validation import cross_val_score

def fit_and_save(estimator, X, y=None, groups=None, scoring=None, cv=None,
                    n_jobs=1, verbose=0, fit_params=None,
                    pre_dispatch='2*n_jobs', parameters=dict(), uuid='', url='http://127.0.0.1:8000'):
    try:
        cv_score = cross_val_score(estimator, X, y, groups, scoring, cv, n_jobs, 
                                   verbose, fit_params, pre_dispatch)
    except Exception as e:
        cv_score = numpy.array([0.])
        parameters['_error'] = '{}: {}'.format(type(e), str(e))
    
    try:
        response = requests.post('{url}/grids/{uuid}/results'.format(url=url, uuid=uuid), 
                      data={'score': round(cv_score.mean(),6), 
                            'gridsearch': uuid, 
                            'cross_validation_scores': cv_score.tolist(),
                            'params': json.dumps(parameters)})
    except requests.exceptions.ConnectionError as e:
        response = None
    if response is None:
        return
    return response
