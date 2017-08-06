import json

from distributed.client import wait
from distributed.deploy.local import LocalCluster
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client as DjangoClient
from django.urls.base import reverse
import numpy
from sklearn import linear_model, ensemble, tree
from sklearn.datasets.base import load_iris
from sklearn.utils.testing import all_estimators

from AnyTimeGridSearchCV.grids.anytime_search import ATGridSearchCV
from AnyTimeGridSearchCV.grids.models import DataSet, GridSearch
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
        self.assertEqual(len(response.data), 12)
        
    def test_estimators_detail(self):
        client = DjangoClient()
        for clf_name, clf in all_estimators():
            response = client.get(reverse('estimator_detail', kwargs={'clf': clf_name}))
            self.assertEqual(200, response.status_code)
            for param_data in response.data:
                self.assertEqual(len(param_data), 3)
                
    def test_dataset_get(self):
        examples_file, label_file = _create_dataset()
        ds, _ = DataSet.objects.get_or_create(name='IRIS', 
                                              examples=SimpleUploadedFile(examples_file.name, examples_file.read()),
                                              labels=SimpleUploadedFile(label_file.name, label_file.read()))
        
        client = DjangoClient()
        response = client.get(reverse('datasets'))
        self.assertEqual(200, response.status_code)
        self.assertEqual(1, len(response.data))
    
    def test_dataset_post(self):
        examples_file, label_file = _create_dataset()
        client = DjangoClient()
        response = client.post(reverse('datasets'), data={'dataset':'IRIS', 'file[0]': examples_file,'file[1]': label_file})
        self.assertEqual(201, response.status_code)
        self.assertEqual(3, len(response.data))
        self.assertEqual(1, DataSet.objects.count())
    
    def test_dataset_grids_get(self):
        reg = linear_model.LinearRegression()
        examples_file, label_file = _create_dataset()
        ds, _ = DataSet.objects.get_or_create(name='IRIS', 
                                              examples=SimpleUploadedFile(examples_file.name, examples_file.read()),
                                              labels=SimpleUploadedFile(label_file.name, label_file.read()))
        
        
        gs_, _ =GridSearch.objects.get_or_create(classifier=reg.__class__.__name__, dataset=ds)
        client = DjangoClient()
        response = client.get(reverse('dataset_grids', kwargs={'name': 'IRIS'}))
        self.assertEqual(200, response.status_code)
        self.assertEqual(1, len(response.data))
        gs_1 = ATGridSearchCV(ensemble.RandomForestClassifier,{'criterion':['gini','entropy'],
                                                         'max_depth':range(1,21),
                                                         'max_features':['auto','log2','sqrt',None]},
                            client_kwargs={'address':LocalCluster()}, dataset=ds.pk, webserver_url=self.live_server_url)
        gs_1.fit()
        response = client.get(reverse('dataset_grids', kwargs={'name': 'IRIS'}))
        self.assertEqual(200, response.status_code)
        self.assertEqual(2, len(response.data))
        
    def test_cvscore_post(self):
        examples_file, label_file = _create_dataset()
        ds, _ = DataSet.objects.get_or_create(name='IRIS', 
                                              examples=SimpleUploadedFile(examples_file.name, examples_file.read()),
                                              labels=SimpleUploadedFile(label_file.name, label_file.read()))
        gs_1 = ATGridSearchCV(ensemble.RandomForestClassifier,{'criterion':['gini','entropy'],
                                                         'max_depth':range(1,21),
                                                         'max_features':['auto','log2','sqrt',None]},
                            client_kwargs={'address':LocalCluster()}, dataset=ds.pk, webserver_url=self.live_server_url)
        from AnyTimeGridSearchCV.grids.anytime_search import fit_and_save
        params = {'criterion': 'gini', 'max_depth': 3, 'max_features': 'log2'}
        res = fit_and_save(ensemble.RandomForestClassifier(**params), 
                           X=numpy.genfromtxt(ds.examples, delimiter=','), 
                           y=numpy.genfromtxt(ds.labels, delimiter=','), 
                           parameters=params, uuid=gs_1._uuid, url= gs_1.webserver_url)
        self.assertEqual(res.status_code, 201)
        
    def test_grids_list_get(self):
        iris = load_iris()
        client = DjangoClient()
        response = client.get(reverse('grids_list'))
        self.assertEqual(200,response.status_code)
        self.assertEqual(0, len(response.data))
        gs1 = ATGridSearchCV(tree.DecisionTreeClassifier,{'criterion':['gini','entropy'],
                                                         'max_depth':range(1,6),
                                                         'max_features':['auto','log2']},
                            client_kwargs={'address':LocalCluster(port=0)}, webserver_url=self.live_server_url)
        gs1.fit(iris.data, iris.target)
        response = client.get(reverse('grids_list'))
        self.assertEqual(200,response.status_code)
        self.assertEqual(1, len(response.data))
        gs2 = ATGridSearchCV(tree.ExtraTreeClassifier,{'criterion':['gini','entropy'],
                                                         'max_depth':range(1,6),
                                                         'max_features':['auto','log2']},
                            client_kwargs={'address':LocalCluster()}, webserver_url=self.live_server_url)
        gs2.fit(iris.data, iris.target)
        response = client.get(reverse('grids_list'))
        self.assertEqual(200, response.status_code)
        self.assertEqual(2, len(response.data))
        
    def test_grids_list_post(self):
        iris = load_iris()
        client = DjangoClient()
        response = client.post(reverse('grids_list'), data={'classifier':'DecisionTreeClassifier'})
        self.assertEqual(201, response.status_code)
        print(response.data)
        gs = ATGridSearchCV(tree.DecisionTreeClassifier,{'criterion':['gini','entropy'],
                                                         'max_depth':range(1,6),
                                                         'max_features':['auto','log2']},
                            client_kwargs={'address':LocalCluster()},
                            uuid=response.data.get('uuid',''),
                            webserver_url=self.live_server_url)
        gs.fit(iris.data, iris.target)
        response = client.get(reverse('grids_list'))
        self.assertEqual(200,response.status_code)
        self.assertEqual(1, len(response.data))
        
    def test_grid_detail(self):
        iris = load_iris()
        client = DjangoClient()
        gs1 = ATGridSearchCV(tree.DecisionTreeClassifier,{'criterion':['gini','entropy'],
                                                         'max_depth':range(1,6),
                                                         'max_features':['auto','log2']},
                            client_kwargs={'address':LocalCluster()},
                            webserver_url=self.live_server_url)
        wait(gs1.fit(iris.data, iris.target))
        response = client.get(reverse('grid_detail', kwargs={'uuid':gs1._uuid}))
        self.assertEqual(200,response.status_code)
        self.assertEqual(response.data['uuid'], str(gs1._uuid))
        
    def test_grid_results_get(self):
        iris = load_iris()
        client = DjangoClient()
        gs1 = ATGridSearchCV(tree.DecisionTreeClassifier,{'criterion':['gini','entropy'],
                                                         'max_depth':range(1,6),
                                                         'max_features':['auto','log2']},
                            client_kwargs={'address':LocalCluster()},
                            webserver_url=self.live_server_url)
        wait(gs1.fit(iris.data, iris.target))
        response = client.get(reverse('grid_results', kwargs={'uuid':gs1._uuid}))
        self.assertEqual(200, response.status_code)
        self.assertEqual(GridSearch.objects.get(uuid=gs1._uuid).results.all().count(), len(response.data))
    
    def test_grid_results_post(self):
        client = DjangoClient()
        gs1 = ATGridSearchCV(tree.DecisionTreeClassifier,{'criterion':['gini','entropy'],
                                                         'max_depth':range(1,5),
                                                         'max_features':['auto','log2']},
                            client_kwargs={'address':LocalCluster()},
                            webserver_url=self.live_server_url)
        response = client.post(reverse('grid_results', kwargs={'uuid':gs1._uuid}), 
                               data={'score': 0.9, 
                                     'gridsearch': gs1._uuid, 
                                     'cross_validation_scores': [0.95, 0.85, 0.9],
                                     'params': json.dumps({'criterion': 'gini', 
                                                'max_depth': 3,
                                                'max_features': 'auto'})})
        self.assertEqual(201, response.status_code)
        
    def test_dataset_grid_results(self):
        examples, labels = _create_dataset()
        ds, _ = DataSet.objects.get_or_create(name='IRIS', 
                                              examples=SimpleUploadedFile(examples.name, examples.read()),
                                              labels=SimpleUploadedFile(labels.name, labels.read()))
        gs = ATGridSearchCV(tree.DecisionTreeClassifier,{'criterion':['gini','entropy'],
                                                         'max_depth':range(1,21),
                                                         'max_features':['auto','log2','sqrt',None]},
                            client_kwargs={'address':LocalCluster()}, dataset=ds.pk,
                            webserver_url=self.live_server_url)
        wait(gs.fit())
        client = DjangoClient()
        response = client.get(reverse('grid_results', kwargs={'uuid':gs._uuid}))
        self.assertEqual(200, response.status_code)
        self.assertEqual(GridSearch.objects.get(uuid=gs._uuid).results.all().count(), len(response.data))