from _io import BytesIO
import uuid
import warnings

from distributed import Client
from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from django.db.models import Max
from scipy.sparse.coo import coo_matrix
from scipy.sparse.csr import csr_matrix
from sklearn.base import clone, is_classifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics.scorer import _check_multimetric_scoring
from sklearn.model_selection._search import GridSearchCV, _check_param_grid, \
    _CVScoreTuple
from sklearn.model_selection._split import check_cv
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.testing import all_estimators
from sklearn.utils.validation import indexable, _num_samples

from AnyTimeGridSearchCV.grids.models import DataSet


ESTIMATORS_DICT = {e[0]:e[1] for e in all_estimators()}

def fit_and_save(estimator, X, y=None, groups=None, scoring=None, cv=None, 
                 n_jobs=1, verbose=0, fit_params=None, pre_dispatch='2*n_jobs', 
                 return_train_score=True, error_score=0., parameters=dict(), uuid='', url='http://127.0.0.1:8000'):

    import json, requests, numpy
    from sklearn.model_selection._validation import cross_validate
    
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    scorers, _ = _check_multimetric_scoring(estimator, scoring=scoring)

    _base_scores = [error_score for _ in range(cv.get_n_splits(X, y, groups))]

    cv_score = {}
    cv_score.update({'train_%s'%s:numpy.array(_base_scores) for s in scorers})
    cv_score.update({'test_%s'%s:numpy.array(_base_scores) for s in scorers})
    cv_score.update({'fit_time':_base_scores, 'score_time': _base_scores})
    
    try:
        estimator = estimator.set_params(**parameters)
        cv_score = cross_validate(estimator, X, y, groups, scorers, cv, n_jobs, verbose, 
                                  fit_params, pre_dispatch, return_train_score)
        error = None
    except Exception as e:
        error = '{}: {}'.format(type(e).__name__, str(e))
    
    try:
        for k, v in cv_score.items():
            if type(v) == type(numpy.array([])):
                cv_score[k] = v.tolist()
        response = requests.post('{url}/grids/{uuid}/results'.format(url=url, uuid=uuid), 
                      data={'gridsearch': uuid, 
                            'params': json.dumps(parameters),
                            'errors': error,
                            'cv_data': json.dumps(cv_score)})
        
    except requests.exceptions.ConnectionError as e: # pragma: no cover
        response = None
    if response is None: # pragma: no cover
        return
    return response

def _convert_clf_param(val):
    if type(val) is dict: 
        return range(int(val['start']), int(val['end'])+1, int(val['skip']))
    elif type(val) is list:
        return [v == 'True' for v in val]
    elif type(val) is str:
        return list(map(str.strip, val.split(',')))

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
    
class ATGridSearchCV(GridSearchCV):
    
    def __init__(self, estimator, param_grid, scoring=None, fit_params=None, 
        n_jobs=1, iid=True, refit=True, cv=None, verbose=0, 
        pre_dispatch='2*n_jobs', error_score=0., 
        return_train_score=True, client_kwargs=settings.DASK_SCHEDULER_PARAMS, uuid='', dataset=None, 
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
        self.dask_client.upload_file(settings.SOURCE_PATH)
        self.dataset, _ = DataSet.objects.get_or_create(pk=dataset) if dataset is not None else (None, None)
        self._uuid = uuid if uuid else GridSearch.objects.create(classifier=type(estimator).__name__, dataset=self.dataset).uuid
        self.webserver_url = webserver_url

    @property
    def best_score_(self):
        from AnyTimeGridSearchCV.grids.models import GridSearch
        if self.refit:
            if self.refit is True: # using 'score' scorer
                return GridSearch.objects.get(uuid=self._uuid).results.filter(scores__scorer='score').aggregate(max_score=Max('scores__score'))['max_score']
            else:
                return GridSearch.objects.get(uuid=self._uuid).results.filter(scores__scorer=self.scoring).aggregate(max_score=Max('scores__score'))['max_score']
        else:
            _scoring = self.scoring if self.scoring is not None else 'score'
            return GridSearch.objects.get(uuid=self._uuid).results.filter(scores__scorer=_scoring).aggregate(max_score=Max('scores__score'))['max_score']
        
    @property
    def best_params_(self):
        from AnyTimeGridSearchCV.grids.models import GridSearch
        return GridSearch.objects.get(uuid=self._uuid).results.filter(scores__score=self.best_score_).first().params
    
    @property
    def best_estimator_(self):
        import numpy
        best_estimator_ = clone(self.estimator.set_params(**self.best_params_))
        if self.refit:
            X = numpy.genfromtxt(self.dataset.examples, delimiter=',')
            y = numpy.genfromtxt(self.dataset.labels, delimiter=',')
            best_estimator_.fit(X, y)
        return best_estimator_
    
    @property
    def cv_results_(self):
        import numpy as np
        from django.contrib.postgres.aggregates.general import ArrayAgg
        from AnyTimeGridSearchCV.grids.models import GridSearch
        
        gridsearch = GridSearch.objects.get(uuid=self._uuid)
        if not gridsearch.results.exists():
            raise NotFittedError("This %(name)s instance is not fitted yet. Call 'fit' with "
               "appropriate arguments before using this method." % {'name': type(self.estimator).__name__})
        self.n_candidates = gridsearch.results.count()
        self.n_splits = len(gridsearch.results.all().first().scores.all().first().test_scores)
        scorers = gridsearch.results.values_list('scores__scorer', flat=True).distinct()
        cv_results_ = gridsearch.results.all().aggregate(fit_time=ArrayAgg('fit_time'), 
                                                         params=ArrayAgg('params'), 
                                                         score_time=ArrayAgg('score_time'))
        scorer_dicts = {scorer: gridsearch.results.filter(scores__scorer=scorer).aggregate(train=ArrayAgg('scores__train_scores'), 
                                                                                 test=ArrayAgg('scores__test_scores'))
                        for scorer in scorers}
        def _store(key_name, array, weights=None, splits=False, rank=False):
            array = np.array(array, dtype=np.float64).reshape(self.n_candidates,
                                                                  self.n_splits)
            if splits:
                for idx, scores in enumerate(array.T):
                    cv_results_['split%d_%s'%(idx, key_name)] = scores
            array_means = np.average(array, axis=1, weights=weights)
            array_stds = np.sqrt(np.average((array -
                                             array_means[:, np.newaxis]) ** 2,
                                            axis=1, weights=weights))
            cv_results_['mean_%s'%key_name] = array_means
            cv_results_['std_%s'%key_name] = array_stds
        
        for operation_time in ['fit_time', 'score_time']:
            _store(operation_time, cv_results_[operation_time])
            del cv_results_[operation_time]
        
        for scorer in scorer_dicts:
            for _set in ['train', 'test']:
                _store('%s_%s'%(_set, scorer), scorer_dicts[scorer][_set], splits=True,
                       weights=self.test_sample_counts if self.iid else None)
        
        return cv_results_
    
    
    @property
    def grid_scores_(self):
        import numpy as np
        if self.multimetric_:
            raise AttributeError("grid_scores_ attribute is not available for"
                                 " multi-metric evaluation.")
        warnings.warn(
            "The grid_scores_ attribute was deprecated in version 0.18"
            " in favor of the more elaborate cv_results_ attribute."
            " The grid_scores_ attribute will not be available from 0.20",
            DeprecationWarning)

        grid_scores = list()

        for i, (params, mean, std) in enumerate(zip(
                self.cv_results_['params'],
                self.cv_results_['mean_test_score'],
                self.cv_results_['std_test_score'])):
            scores = np.array(list(self.cv_results_['split%d_test_score'
                                                    % s][i]
                                   for s in range(self.n_splits)),
                              dtype=np.float64)
            grid_scores.append(_CVScoreTuple(params, mean, scores))

        return grid_scores
    
    def fit(self, X=None, y=None, groups=None, **fit_params):
        """Run fit with all sets of parameters.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of the estimator
        """
        if self.fit_params is not None:
            warnings.warn('"fit_params" as a constructor argument was '
                          'deprecated in version 0.19 and will be removed '
                          'in version 0.21. Pass fit parameters to the '
                          '"fit" method instead.', DeprecationWarning)
            if fit_params:
                warnings.warn('Ignoring fit_params passed as a constructor '
                              'argument in favor of keyword arguments to '
                              'the "fit" method.', RuntimeWarning)
            else:
                fit_params = self.fit_params
        if X is None and self.dataset is None:
            raise NoDatasetError('No data provided')
        import numpy, six
        from tempfile import TemporaryFile

        if self.dataset is not None and X is None:
            X = numpy.genfromtxt(self.dataset.examples, delimiter=',')
            y = numpy.genfromtxt(self.dataset.labels, delimiter=',')
            
        if self.dataset is None and X is not None:
            examples_file, label_file = TemporaryFile(), TemporaryFile()
            try:
                numpy.savetxt(examples_file, X if type(X) not in [coo_matrix, csr_matrix] else X.toarray(), delimiter=',')
            except TypeError:
                X.tofile(examples_file, sep=',')
            examples_file.seek(0)
            if y is not None:
                try:
                    numpy.savetxt(label_file, y if type(y) not in [coo_matrix, csr_matrix] else y.toarray(), delimiter=',')
                except TypeError:
                    y.tofile(label_file, sep=',')
                label_file.seek(0)
            else:
                numpy.savetxt(label_file, [], delimiter=',')
                label_file.seek(0)   
            self.dataset = DataSet.objects.create(name=str(uuid.uuid4()), 
                                              examples=SimpleUploadedFile('examples.csv', examples_file.read()),
                                              labels=SimpleUploadedFile('labels.csv', label_file.read()))
            
        estimator = self.estimator
        cv = check_cv(self.cv, y, classifier=is_classifier(estimator))

        scorers, self.multimetric_ = _check_multimetric_scoring(
            self.estimator, scoring=self.scoring)

        if self.multimetric_:
            if self.refit is not False and (
                    not isinstance(self.refit, six.string_types) or
                    # This will work for both dict / list (tuple)
                    self.refit not in scorers):
                raise ValueError("For multi-metric scoring, the parameter "
                                 "refit must be set to a scorer key "
                                 "to refit an estimator with the best "
                                 "parameter setting on the whole data and "
                                 "make the best_* attributes "
                                 "available for that metric. If this is not "
                                 "needed, refit should be set to False "
                                 "explicitly. %r was passed." % self.refit)
            else:
                refit_metric = self.refit
        else:
            refit_metric = 'score'

        X, y, groups = indexable(X, y, groups)
        n_splits = cv.get_n_splits(X, y, groups)
        # Regenerate parameter iterable for each fit
        candidate_params = list(self._get_param_iterator())
        n_candidates = len(candidate_params)
        if self.verbose > 0:
            print("Fitting {0} folds for each of {1} candidates, totaling"
                  " {2} fits".format(n_splits, n_candidates,
                                     n_candidates * n_splits))
            
        self.scorer_ = scorers if self.multimetric_ else scorers['score']
        
        self.test_sample_counts = [_num_samples(_safe_split(estimator, X, y, test, train)[0]) 
                                   for train, test in cv.split(X,y,groups)]
        

        return [self.dask_client.submit(fit_and_save, self.estimator, 
                                               X, y, groups= groups, scoring=scorers, cv=self.cv, 
                                               n_jobs=self.n_jobs, verbose=self.verbose, fit_params=fit_params, 
                                               pre_dispatch=self.pre_dispatch, parameters=parameters, 
                                               error_score=self.error_score, uuid=self._uuid,
                                               url=self.webserver_url) for parameters in candidate_params]
        
    def _check_is_fitted(self, method_name):
        from AnyTimeGridSearchCV.grids.models import GridSearch
        if not self.refit:
            raise NotFittedError('This %s instance was initialized '
                                 'with refit=False. %s is '
                                 'available only after refitting on the best '
                                 'parameters. You can refit an estimator '
                                 'manually using the ``best_parameters_`` '
                                 'attribute'
                                 % (type(self).__name__, method_name))
        else:
            gridsearch = GridSearch.objects.get(uuid=self._uuid)
            if not gridsearch.results.exists():
                raise NotFittedError("This %(name)s instance is not fitted yet. Call 'fit' with "
                   "appropriate arguments before using this method." % {'name': type(self.estimator).__name__})
                
    def score(self, X, y=None):
        self._check_is_fitted('score')
        if self.scorer_ is None:
            raise ValueError("No score function explicitly defined, "
                             "and the estimator doesn't provide one %s"
                             % self.best_estimator_)
        score = self.scorer_[self.refit] if self.multimetric_ else self.scorer_
        best_estimator = self.best_estimator_
        best_estimator.fit(X,y)
        return score(best_estimator, X, y)
        
