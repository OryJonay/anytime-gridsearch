from _io import BytesIO
import json

from distributed.client import wait
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client as DjangoClient
from django.urls.base import reverse
import numpy
from sklearn import linear_model, ensemble, tree
from sklearn.datasets.base import load_iris, load_breast_cancer
from sklearn.utils.testing import all_estimators

from AnyTimeGridSearchCV.grids.anytime_search import ATGridSearchCV, \
    fit_and_save
from AnyTimeGridSearchCV.grids.models import DataSet, GridSearch, CVResult
from AnyTimeGridSearchCV.grids.tests import AbstractGridsTestCase, \
    _create_dataset


class TestViews(AbstractGridsTestCase):
    
    def test_list_estimators(self):
        client = DjangoClient()
        response = client.get(reverse('estimators_list'))
        self.assertEqual(200, response.status_code)
        self.assertEqual(len(response.data), len(all_estimators()))
    
    def test_estimator_detail(self):
        client = DjangoClient()
        response = client.get(reverse('estimator_detail', kwargs={'clf': 'DecisionTreeClassifier'}))
        self.assertEqual(200, response.status_code)
        self.assertEqual(len(response.data), len(tree.DecisionTreeClassifier().get_params()))
        
    def test_estimator_detail_bad(self):
        client = DjangoClient()
        response = client.get(reverse('estimator_detail', kwargs={'clf': 'alice and bob'}))
        self.assertEqual(200, response.status_code)
        self.assertEqual(len(response.data), 3)
        
    def test_estimators_detail(self):
        client = DjangoClient()
        for clf_name, clf in all_estimators():
            response = client.get(reverse('estimator_detail', kwargs={'clf': clf_name}))
            self.assertEqual(200, response.status_code)
            for param_data in response.data:
                self.assertEqual(len(param_data), 3)
                
    def test_dataset_get(self):
        examples_file, label_file = _create_dataset()
        ds, _ = DataSet.objects.get_or_create(name='TEST', 
                                              examples=SimpleUploadedFile(examples_file.name, examples_file.read()),
                                              labels=SimpleUploadedFile(label_file.name, label_file.read()))
        
        client = DjangoClient()
        response = client.get(reverse('datasets'))
        self.assertEqual(200, response.status_code)
        self.assertEqual(1, len(response.data))
    
    def test_dataset_post_success(self):
        examples_file, label_file = _create_dataset()
        client = DjangoClient()
        response = client.post(reverse('datasets'), data={'name':'TEST', 'examples': examples_file, 'labels': label_file})
        self.assertEqual(201, response.status_code)
        self.assertEqual(3, len(response.data))
        self.assertEqual(1, DataSet.objects.count())
        
    def test_dataset_post_duplicate_name(self):
        examples_file, label_file = _create_dataset()
        client = DjangoClient()
        response = client.post(reverse('datasets'), data={'name':'TEST', 'examples': examples_file, 'labels': label_file})
        self.assertEqual(201, response.status_code)
        response = client.post(reverse('datasets'), data={'name':'TEST', 'examples': examples_file, 'labels': label_file})
        self.assertEqual(400, response.status_code)
        self.assertEqual(b'"Name already exists"', response.content)
    
    def test_dataset_post_no_dataset(self):
        client = DjangoClient()
        response = client.post(reverse('datasets'), data={})
        self.assertEqual(400, response.status_code)
        self.assertEqual(b'"Missing dataset name"', response.content)
        
    def test_dataset_post_blank_name(self):
        examples_file, label_file = _create_dataset()
        client = DjangoClient()
        response = client.post(reverse('datasets'), data={'name':'', 'examples': examples_file, 'labels': label_file})
        self.assertEqual(400, response.status_code)
        self.assertEqual(b'"Missing dataset name"', response.content)
        
    def test_dataset_post_missing_examples(self):
        examples_file, label_file = _create_dataset()
        client = DjangoClient()
        response = client.post(reverse('datasets'), data={'name':'TEST', 'labels': label_file})
        self.assertEqual(400, response.status_code)
        self.assertEqual(b'"Missing dataset files"', response.content)
        
    def test_dataset_post_missing_labels(self):
        examples_file, label_file = _create_dataset()
        client = DjangoClient()
        response = client.post(reverse('datasets'), data={'name':'TEST', 'examples': examples_file})
        self.assertEqual(400, response.status_code)
        self.assertEqual(b'"Missing dataset files"', response.content)
        
    def test_dataset_post_examples_bad_name(self):
        examples_file, label_file = _create_dataset()
        examples_file.name = 'EXAMPLES.csv'
        client = DjangoClient()
        response = client.post(reverse('datasets'), data={'name':'TEST', 'examples': examples_file, 'labels': label_file})
        self.assertEqual(400, response.status_code)
        self.assertEqual(b'"Bad name of examples file"', response.content)
    
    def test_dataset_post_labels_bad_name(self):
        examples_file, label_file = _create_dataset()
        label_file.name = 'EXAMPLES.csv'
        client = DjangoClient()
        response = client.post(reverse('datasets'), data={'name':'TEST', 'examples': examples_file, 'labels': label_file})
        self.assertEqual(400, response.status_code)
        self.assertEqual(b'"Bad name of labels file"', response.content)
        
    def test_dataset_post_dataset_length_mismatch(self):
        examples_file, label_file = BytesIO(), BytesIO()
        examples_file.name = 'examples.csv'
        label_file.name = 'labels.csv'
        iris = load_iris()
        breast_cancer = load_breast_cancer()
        numpy.savetxt(examples_file, iris.data, delimiter=',')
        numpy.savetxt(label_file, breast_cancer.target, delimiter=',')
        examples_file.seek(0), label_file.seek(0)
        client = DjangoClient()
        response = client.post(reverse('datasets'), data={'name':'TEST', 'examples': examples_file, 'labels': label_file})
        self.assertEqual(400, response.status_code)
        self.assertEqual(b'"Examples and labels are not the same length"', response.content)
    
    def test_atgridsearch_post(self):
        examples_file, label_file = _create_dataset()
        ds, _ = DataSet.objects.get_or_create(name='TEST', 
                                              examples=SimpleUploadedFile(examples_file.name, examples_file.read()),
                                              labels=SimpleUploadedFile(label_file.name, label_file.read()))
        post_data = {'clf':tree.DecisionTreeClassifier.__name__, 'dataset':ds.name}
        post_data['args'] = {'criterion': 'gini, entropy',
                             'max_features': {'start': 5, 'end': 10, 'skip': 1},
                             'presort': ['True', 'False']}
        
        response = DjangoClient().post(reverse('gridsearch_create'), json.dumps(post_data), content_type="application/json")
        self.assertEqual(201, response.status_code, response.data)
        
    def test_atgridsearch_post_floats(self):
        examples_file, label_file = _create_dataset()
        ds, _ = DataSet.objects.get_or_create(name='TEST', 
                                              examples=SimpleUploadedFile(examples_file.name, examples_file.read()),
                                              labels=SimpleUploadedFile(label_file.name, label_file.read()))
        post_data = {'clf':tree.DecisionTreeClassifier.__name__, 'dataset':ds.name}
        post_data['args'] = {'max_features': {'start': 5.5, 'end': 12.5, 'skip': 3}}
        
        response = DjangoClient().post(reverse('gridsearch_create'), json.dumps(post_data), content_type="application/json")
        self.assertEqual(201, response.status_code, response.data)
        
    def test_atgridsearch_post_no_dataset(self):
        post_data = {'clf':tree.DecisionTreeClassifier.__name__, 'dataset':'TEST'}
        post_data['args'] = {'criterion': 'gini, entropy',
                             'max_features': {'start': 5, 'end': 10, 'skip': 1}}
        
        response = DjangoClient().post(reverse('gridsearch_create'), json.dumps(post_data), content_type="application/json")
        self.assertEqual(400, response.status_code)
        self.assertEqual(b'"No DataSet named TEST"', response.content)
        
    def test_atgridsearch_post_no_clf(self):
        examples_file, label_file = _create_dataset()
        ds, _ = DataSet.objects.get_or_create(name='TEST', 
                                              examples=SimpleUploadedFile(examples_file.name, examples_file.read()),
                                              labels=SimpleUploadedFile(label_file.name, label_file.read()))
        post_data = {'clf':'Tree', 'dataset':ds.name}
        post_data['args'] = {'criterion': 'gini, entropy',
                             'max_features': {'start': 5, 'end': 10, 'skip': 1}}
        
        response = DjangoClient().post(reverse('gridsearch_create'), json.dumps(post_data), content_type="application/json")
        self.assertEqual(400, response.status_code)
        self.assertEqual(b'"No sklearn classifier named Tree"', response.content)
        
    def test_dataset_grids_get(self):
        reg = linear_model.LinearRegression()
        examples_file, label_file = _create_dataset()
        ds, _ = DataSet.objects.get_or_create(name='TEST', 
                                              examples=SimpleUploadedFile(examples_file.name, examples_file.read()),
                                              labels=SimpleUploadedFile(label_file.name, label_file.read()))
        
        
        gs_, _ =GridSearch.objects.get_or_create(classifier=reg.__class__.__name__, dataset=ds)
        client = DjangoClient()
        response = client.get(reverse('dataset_grids', kwargs={'name': 'TEST'}))
        self.assertEqual(200, response.status_code)
        self.assertEqual(1, len(response.data))
        gs_1 = ATGridSearchCV(ensemble.RandomForestClassifier(),{'criterion':['gini','entropy'],
                                                         'max_depth':range(1,21),
                                                         'max_features':['auto','log2','sqrt',None]},
                            dataset=ds.pk, webserver_url=self.live_server_url)
        wait(gs_1.fit())
        response = client.get(reverse('dataset_grids', kwargs={'name': 'TEST'}))
        self.assertEqual(200, response.status_code)
        self.assertEqual(2, len(response.data))
        
    def test_cvscore_post(self):
        examples_file, label_file = _create_dataset()
        ds, _ = DataSet.objects.get_or_create(name='TEST', 
                                              examples=SimpleUploadedFile(examples_file.name, examples_file.read()),
                                              labels=SimpleUploadedFile(label_file.name, label_file.read()))
        gs_1 = ATGridSearchCV(ensemble.RandomForestClassifier(),{'criterion':['gini','entropy'],
                                                         'max_depth':range(1,21),
                                                         'max_features':['auto','log2','sqrt',None]},
                            dataset=ds.pk, webserver_url=self.live_server_url)
        params = {'criterion': 'gini', 'max_depth': 3, 'max_features': 'log2'}
        res = fit_and_save(ensemble.RandomForestClassifier(**params), 
                           X=numpy.genfromtxt(ds.examples, delimiter=','), 
                           y=numpy.genfromtxt(ds.labels, delimiter=','), 
                           parameters=params, uuid=gs_1._uuid, url= gs_1.webserver_url)
        self.assertEqual(res.status_code, 201)
        self.assertEqual(CVResult.objects.count(), 1)
        self.assertEqual(CVResult.objects.get(id=res.json()['id']).scores.all().count(), 1)
        
    def test_cvscore_post_multimetric(self):
        from sklearn.metrics import make_scorer, accuracy_score, jaccard_similarity_score
        examples_file, label_file = _create_dataset()
        ds, _ = DataSet.objects.get_or_create(name='TEST', 
                                              examples=SimpleUploadedFile(examples_file.name, examples_file.read()),
                                              labels=SimpleUploadedFile(label_file.name, label_file.read()))
        gs_1 = ATGridSearchCV(tree.DecisionTreeClassifier,{'criterion':['gini','entropy'],
                                                         'max_depth':range(1,21),
                                                         'max_features':['auto','log2','sqrt',None]},
                            dataset=ds.pk, webserver_url=self.live_server_url,
                            scoring={'Accuracy': make_scorer(accuracy_score), 
                                     'Jaccard':make_scorer(jaccard_similarity_score)})
        params = {'criterion': 'gini', 'max_depth': 3, 'max_features': 'log2'}
        res = fit_and_save(tree.DecisionTreeClassifier(**params), 
                           X=numpy.genfromtxt(ds.examples, delimiter=','), 
                           y=numpy.genfromtxt(ds.labels, delimiter=','), 
                           parameters=params, uuid=gs_1._uuid, url= gs_1.webserver_url,
                           scoring={'Accuracy': make_scorer(accuracy_score), 
                                    'Jaccard':make_scorer(jaccard_similarity_score)})
        self.assertEqual(res.status_code, 201)
        self.assertEqual(CVResult.objects.count(), 1)
        self.assertEqual(CVResult.objects.get(id=res.json()['id']).scores.all().count(), 2)
    
    def test_cvscore_post_bad_args(self):
        examples_file, label_file = _create_dataset()
        ds, _ = DataSet.objects.get_or_create(name='TEST', 
                                              examples=SimpleUploadedFile(examples_file.name, examples_file.read()),
                                              labels=SimpleUploadedFile(label_file.name, label_file.read()))
        gs_1 = ATGridSearchCV(ensemble.RandomForestClassifier(),{'criterion':['gini','entropy'],
                                                         'max_depth':range(1,21),
                                                         'max_features':['auto','log2','sqrt',None]},
                            dataset=ds.pk, webserver_url=self.live_server_url)
        params = {'criterion': 'gini', 'max_depth': 3, 'max_features': 1000}
        res = fit_and_save(ensemble.RandomForestClassifier(**params), 
                           X=numpy.genfromtxt(ds.examples, delimiter=','), 
                           y=numpy.genfromtxt(ds.labels, delimiter=','), 
                           parameters=params, uuid=gs_1._uuid, url= gs_1.webserver_url)
        self.assertEqual(res.status_code, 201)
        self.assertEqual(res.json()['fit_time'], [0., 0., 0.])
        
    def test_cvscore_post_no_server(self):
        examples_file, label_file = _create_dataset()
        ds, _ = DataSet.objects.get_or_create(name='TEST', 
                                              examples=SimpleUploadedFile(examples_file.name, examples_file.read()),
                                              labels=SimpleUploadedFile(label_file.name, label_file.read()))
        gs_1 = ATGridSearchCV(ensemble.RandomForestClassifier(),{'criterion':['gini','entropy'],
                                                         'max_depth':range(1,21),
                                                         'max_features':['auto','log2','sqrt',None]},
                            dataset=ds.pk)
        params = {'criterion': 'gini', 'max_depth': 3, 'max_features': 'auto'}
        res = fit_and_save(ensemble.RandomForestClassifier(**params), 
                           X=numpy.genfromtxt(ds.examples, delimiter=','), 
                           y=numpy.genfromtxt(ds.labels, delimiter=','), 
                           parameters=params, uuid=gs_1._uuid, url= gs_1.webserver_url)
        if hasattr(res, 'status_code'): # pragma: no cover
            self.assertEqual(res.status_code, 404)
        else:
            self.assertIsNone(res)
    def test_grids_list_get(self):
        iris = load_iris()
        client = DjangoClient()
        response = client.get(reverse('grids_list'))
        self.assertEqual(200,response.status_code)
        self.assertEqual(0, len(response.data))
        gs1 = ATGridSearchCV(tree.DecisionTreeClassifier(),{'criterion':['gini','entropy'],
                                                         'max_depth':range(1,6),
                                                         'max_features':['auto','log2']},
                            webserver_url=self.live_server_url)
        wait(gs1.fit(iris.data, iris.target))
        response = client.get(reverse('grids_list'))
        self.assertEqual(200,response.status_code)
        self.assertEqual(1, len(response.data))
        gs2 = ATGridSearchCV(tree.ExtraTreeClassifier(),{'criterion':['gini','entropy'],
                                                         'max_depth':range(1,6),
                                                         'max_features':['auto','log2']},
                            webserver_url=self.live_server_url)
        wait(gs2.fit(iris.data, iris.target))
        response = client.get(reverse('grids_list'))
        self.assertEqual(200, response.status_code)
        self.assertEqual(2, len(response.data))
        
    def test_grids_list_post(self):
        iris = load_iris()
        client = DjangoClient()
        response = client.post(reverse('grids_list'), data={'classifier':'DecisionTreeClassifier'})
        self.assertEqual(201, response.status_code)
        gs = ATGridSearchCV(tree.DecisionTreeClassifier(),{'criterion':['gini','entropy'],
                                                         'max_depth':range(1,6),
                                                         'max_features':['auto','log2']},
                            uuid=response.data.get('uuid',''),
                            webserver_url=self.live_server_url)
        wait(gs.fit(iris.data, iris.target))
        response = client.get(reverse('grids_list'))
        self.assertEqual(200,response.status_code)
        self.assertEqual(1, len(response.data))
        
    def test_grid_detail(self):
        iris = load_iris()
        client = DjangoClient()
        gs1 = ATGridSearchCV(tree.DecisionTreeClassifier(),{'criterion':['gini','entropy'],
                                                         'max_depth':range(1,6),
                                                         'max_features':['auto','log2']},
                            webserver_url=self.live_server_url)
        wait(gs1.fit(iris.data, iris.target))
        response = client.get(reverse('grid_detail', kwargs={'uuid':gs1._uuid}))
        self.assertEqual(200,response.status_code)
        self.assertEqual(response.data['uuid'], str(gs1._uuid))
        
    def test_grid_results_get(self):
        iris = load_iris()
        client = DjangoClient()
        gs1 = ATGridSearchCV(tree.DecisionTreeClassifier(),{'criterion':['gini','entropy'],
                                                         'max_depth':range(1,6),
                                                         'max_features':['auto','log2']},
                            webserver_url=self.live_server_url)
        wait(gs1.fit(iris.data, iris.target))
        response = client.get(reverse('grid_results', kwargs={'uuid':gs1._uuid}))
        self.assertEqual(200, response.status_code)
        self.assertEqual(GridSearch.objects.get(uuid=gs1._uuid).results.all().count(), len(response.data))
    
    def test_dataset_grid_results(self):
        examples, labels = _create_dataset()
        ds, _ = DataSet.objects.get_or_create(name='TEST', 
                                              examples=SimpleUploadedFile(examples.name, examples.read()),
                                              labels=SimpleUploadedFile(labels.name, labels.read()))
        gs = ATGridSearchCV(tree.DecisionTreeClassifier(),{'criterion':['gini','entropy'],
                                                         'max_depth':range(1,21),
                                                         'max_features':['auto','log2','sqrt',None]},
                            dataset=ds.pk,
                            webserver_url=self.live_server_url)
        wait(gs.fit())
        client = DjangoClient()
        response = client.get(reverse('grid_results', kwargs={'uuid':gs._uuid}))
        self.assertEqual(200, response.status_code)
        self.assertEqual(GridSearch.objects.get(uuid=gs._uuid).results.all().count(), len(response.data))