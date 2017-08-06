import json

from distributed import Client
from django.db.models import Max
import numpy, requests
from sklearn.base import clone
from sklearn.model_selection._search import GridSearchCV, _check_param_grid
from sklearn.model_selection._validation import cross_val_score
from sklearn.utils.testing import all_estimators

ESTIMATORS_DICT = {e[0]:e[1] for e in all_estimators()}

class NoDatasetError(ValueError, AttributeError):
    """Exception class to raise if ATGridSearchCV is used for fitting without a dataset.

    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.

    Examples
    --------
    >>> try:
    ...     ATGridSearchCV().fit()
    ... except NoDatasetError as e:
    ...     print(repr(e))
    ...                        # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    NotFittedError('This LinearSVC instance is not fitted yet',)
    """

def fit_and_save(estimator, X, y=None, groups=None, scoring=None, cv=None,
                    n_jobs=1, verbose=0, fit_params=None,
                    pre_dispatch='2*n_jobs', parameters=dict(), uuid='', url='http://127.0.0.1:8000'):
    try:
        cv_score = cross_val_score(estimator, X, y, groups, scoring, cv, n_jobs, 
                                   verbose, fit_params, pre_dispatch)
    except Exception as e:
        print('Not created because of uncaught exception in cv_score, type and reason: {} {}'.format(type(e), str(e)))
        cv_score = [0.]
        
    response = requests.post('{url}/grids/{uuid}/results'.format(url=url, uuid=uuid), 
                  data={'score': round(cv_score.mean(),6), 
                        'gridsearch': uuid, 
                        'cross_validation_scores': cv_score.tolist(),
                        'params': json.dumps(parameters)})
    if response.status_code != 201:
        print('Not created, status and reason: {status} {reason}'.format(status=response.status_code,
                                                                         reason=response.content))
    return response

class ATGridSearchCV(GridSearchCV):
    
    def __init__(self, estimator, param_grid, scoring=None, fit_params=None, 
        n_jobs=1, iid=True, refit=False, cv=None, verbose=0, 
        pre_dispatch='2*n_jobs', error_score='raise', 
        return_train_score=True, client_kwargs=dict(), uuid='', dataset=None, 
        webserver_url='http://127.0.0.1:8000'):
        super(GridSearchCV, self).__init__(
            estimator=estimator, scoring=scoring, fit_params=fit_params,
            n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)
        from AnyTimeGridSearchCV.grids.models import GridSearch,  DataSet
        self.param_grid = param_grid
        _check_param_grid(param_grid)
        self.dask_client = Client(silence_logs=100, **client_kwargs)
        self.dataset, _ = DataSet.objects.get_or_create(pk=dataset) if dataset is not None else (None, None)
        self._uuid = uuid if uuid else GridSearch.objects.create(classifier=estimator.__name__, dataset=self.dataset).uuid
        self.webserver_url = webserver_url

    def _fit(self, X, y, groups, parameter_iterable):
        """Actual fitting,  performing the search over parameters."""

        if self.dataset is not None and X is None:
            X = numpy.genfromtxt(self.dataset.examples, delimiter=',')
            y = numpy.genfromtxt(self.dataset.labels, delimiter=',')
        
        return [self.dask_client.submit(fit_and_save, clone(self.estimator(**parameters)), 
                                               X, y, groups= groups, scoring=self.scoring, cv=self.cv, 
                                               n_jobs=self.n_jobs, verbose=self.verbose, fit_params=self.fit_params, 
                                               pre_dispatch=self.pre_dispatch, parameters=parameters, 
                                               uuid=self._uuid,
                                               url=self.webserver_url) for parameters in parameter_iterable]
            
    @property
    def best_score_(self):
        from AnyTimeGridSearchCV.grids.models import GridSearch
        return GridSearch.objects.get(uuid=self._uuid).results.aggregate(max_score=Max('score'))['max_score']
    
    @property
    def best_params_(self):
        from AnyTimeGridSearchCV.grids.models import GridSearch
        return GridSearch.objects.get(uuid=self._uuid).results.filter(score=self.best_score_).first().params
    
    @property
    def best_estimator_(self):
        return clone(self.estimator(**self.best_params_))
    
    def fit(self, X=None, y=None, groups=None):
        if X is None and self.dataset is None:
            raise NoDatasetError('No data provided')
        return GridSearchCV.fit(self, X, y=y, groups=groups)
