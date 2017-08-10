'''
Created on Aug 10, 2017

@author: Ory Jonay
'''
import json
import logging
import os

import numpy
import requests
from sklearn.model_selection._validation import cross_val_score
  
  
logging.getLogger(__name__)
logging.basicConfig(filename=os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                          '..',
                                          'fit.log'), 
                    level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)s:%(message)s', 
                    datefmt='%m/%d/%Y %I:%M:%S %p')


def fit_and_save(estimator, X, y=None, groups=None, scoring=None, cv=None,
                    n_jobs=1, verbose=0, fit_params=None,
                    pre_dispatch='2*n_jobs', parameters=dict(), uuid='', url='http://127.0.0.1:8000'):
    try:
        logging.debug('Trying to use cv_score')
        cv_score = cross_val_score(estimator, X, y, groups, scoring, cv, n_jobs, 
                                   verbose, fit_params, pre_dispatch)
        logging.info('cv_score success %d', round(cv_score.mean(),6))
    except Exception as e:
        logging.error('Not created because of uncaught exception in cv_score, type and reason: {} {}'.format(type(e), str(e)))
        cv_score = numpy.array([0.])
        parameters['_error'] = '{}: {}'.format(type(e), str(e))
    
    logging.debug('Trying to post results back to server')
    try:
        response = requests.post('{url}/grids/{uuid}/results'.format(url=url, uuid=uuid), 
                      data={'score': round(cv_score.mean(),6), 
                            'gridsearch': uuid, 
                            'cross_validation_scores': cv_score.tolist(),
                            'params': json.dumps(parameters)})
    except requests.exceptions.ConnectionError as e:
        logging.error('Exception in post: {type} {exc}'.format(type=type(e), exc=str(e)))
        response = None
    if response is None:
        return
#     logging.info('POST request status code %d', response.status_code)
#     if response.status_code != 201:
#         logging.error('Not created, status and reason: {status} {reason}'.format(status=response.status_code,
#                                                                          reason=response.content))
    return response
