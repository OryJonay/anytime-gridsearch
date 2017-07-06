from io import BytesIO
import json
import shutil
import unittest

from distributed import Client, LocalCluster, wait
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management import call_command
from django.db.models.expressions import RawSQL
from django.test import LiveServerTestCase, Client as DjangoClient
from django.urls.base import reverse
import numpy
from sklearn import linear_model, ensemble
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection._validation import cross_val_score

from AnyTimeGridSearchCV.grids.anytime_search import ATGridSearchCV, \
    NoDatasetError
from AnyTimeGridSearchCV.grids.models import GridSearch, CVResult, DataSet


def _create_dataset():
        examples_file, label_file = BytesIO(), BytesIO()
        examples_file.name = 'examples.csv'
        label_file.name = 'labels.csv'
        iris = load_iris()
        numpy.savetxt(examples_file, iris.data, delimiter=',')
        numpy.savetxt(label_file, iris.target, delimiter=',')
        examples_file.seek(0), label_file.seek(0)
        return examples_file, label_file

class TestModels(LiveServerTestCase):
    
    def setUp(self):
        super(TestModels, self).setUp()
        try:
            shutil.rmtree('media/datasets/IRIS/train/')
            shutil.rmtree('media/datasets/IRIS/test/')
        except FileNotFoundError:
            pass
    
    def test_dataset_model_single_file(self):
        examples_file , label_file = BytesIO() , BytesIO()
        examples_file.name = 'examples.csv'
        label_file.name = 'labels.csv'
        iris = load_iris()
        numpy.savetxt(examples_file, iris.data, delimiter=',')
        numpy.savetxt(label_file, iris.target, delimiter=',')
        examples_file.seek(0) , label_file.seek(0)
        ds, _ = DataSet.objects.get_or_create(name='IRIS', 
                                              examples=SimpleUploadedFile(examples_file.name, examples_file.read()),
                                              labels=SimpleUploadedFile(label_file.name, label_file.read()))
        self.assertEqual('datasets/IRIS/train/examples.csv', ds.examples.name)
        self.assertEqual('datasets/IRIS/test/labels.csv', ds.labels.name)
        loaded_train = numpy.genfromtxt(ds.examples, delimiter=',')
        loaded_labels = numpy.genfromtxt(ds.labels, delimiter=',')
        self.assertTrue(numpy.array_equal(loaded_train, iris.data))
        self.assertTrue(numpy.array_equal(loaded_labels, iris.target))
            
    def test_grid_search_model_creation(self):
        reg = linear_model.LinearRegression()
        examples_file , label_file = BytesIO() , BytesIO()
        examples_file.name = 'examples.csv'
        label_file.name = 'labels.csv'
        iris = load_iris()
        numpy.savetxt(examples_file, iris.data, delimiter=',')
        numpy.savetxt(label_file, iris.target, delimiter=',')
        examples_file.seek(0) , label_file.seek(0)
        ds, _ = DataSet.objects.get_or_create(name='IRIS', 
                                              examples=SimpleUploadedFile(examples_file.name, examples_file.read()),
                                              labels=SimpleUploadedFile(label_file.name, label_file.read()))
        GridSearch.objects.get_or_create(classifier=reg.__class__.__name__, dataset=ds)
    
    def test_grid_search_result_model_creation(self):
        reg = linear_model.LinearRegression()
        _gs , _ = GridSearch.objects.get_or_create(classifier=reg.__class__.__name__)
        reg.fit ([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
        _score = reg.predict([[0,0.1],[0,0.5]]).mean()
        CVResult.objects.get_or_create(score=_score,params=reg.get_params(),gridsearch=_gs)
        
    def test_grid_search_sorting_by_inner_params(self):
        reg2 = linear_model.Ridge(alpha=0.5)
        reg3 = linear_model.Ridge(alpha=0.8)
        _gs , _ = GridSearch.objects.get_or_create(classifier=reg2.__class__.__name__)
        r2 , _ = CVResult.objects.get_or_create(score=0.7, params=reg2.get_params(), gridsearch=_gs)
        r3 , _ = CVResult.objects.get_or_create(score=0.9, params=reg3.get_params(), gridsearch=_gs)
        self.assertListEqual([r2,r3],list(CVResult.objects.filter(**{'params__alpha__isnull':False})
                                          .order_by(RawSQL("params->>%s", ("alpha",)))))
        self.assertListEqual([r3,r2],list(CVResult.objects.filter(**{'params__alpha__isnull':False})
                                          .order_by(RawSQL("params->>%s", ("alpha",))))[::-1])
        self.assertListEqual([r2,r3], list(_gs.order_results_by('alpha')))

class TestValidation(LiveServerTestCase):
    def setUp(self):
        super(TestValidation, self).setUp()
        GridSearch.objects.all().delete()
        DataSet.objects.all().delete()
        try:
            shutil.rmtree('media/datasets/IRIS/train/')
            shutil.rmtree('media/datasets/IRIS/test/')
        except FileNotFoundError:
            pass
    
    def test_cv_saving_explicit(self):
        iris = load_iris()
        reg = tree.DecisionTreeClassifier()
        _gs , _ = GridSearch.objects.get_or_create(classifier=reg.__class__.__name__)
        cv_score = cross_val_score(reg,iris.data,iris.target)
        reg2 = tree.DecisionTreeClassifier(criterion='entropy')
        cv_score2 = cross_val_score(reg2,iris.data,iris.target)
        reg3 = tree.DecisionTreeClassifier(max_features='log2')
        cv_score3 = cross_val_score(reg3,iris.data,iris.target)
        CVResult.objects.create(score=cv_score.mean(),params=reg.get_params(),gridsearch=_gs,
                                       cross_validation_scores = cv_score.tolist())
        CVResult.objects.create(score=cv_score2.mean(),params=reg.get_params(),gridsearch=_gs,
                                       cross_validation_scores = cv_score2.tolist())
        CVResult.objects.create(score=cv_score3.mean(),params=reg.get_params(),gridsearch=_gs,
                                       cross_validation_scores = cv_score3.tolist())
        self.assertEqual(3, _gs.results.count())
    
    def test_dask_cv_single(self):
        test_cluster = LocalCluster(1)
        test_client = Client(test_cluster)
        iris = load_iris()
        reg = tree.DecisionTreeClassifier()
        cv_score = test_client.submit(cross_val_score,reg,iris.data,iris.target)
        self.assertGreater(cv_score.result().mean(), 0)
        test_cluster.scale_up(4)
        _cv_results = {'reg_%i':test_client.submit(cross_val_score,
                                                  tree.DecisionTreeClassifier(min_samples_leaf=i),iris.data,iris.target)
                      for i in range(5)} 
        cv_results = test_client.gather(list(_cv_results.values()))
        for cv_result in cv_results:
            self.assertGreaterEqual(cv_result.mean(), 0)
            
    def test_ATGridSearchCV_no_dataset(self):
        iris = load_iris()
        grid_size = 2*20*4
        gs = ATGridSearchCV(tree.DecisionTreeClassifier,{'criterion':['gini','entropy'],
                                                         'max_depth':range(1,21),
                                                         'max_features':['auto','log2','sqrt',None]},
                            client_kwargs={'address':LocalCluster()},
                            webserver_url=self.live_server_url)
        wait(gs.fit(iris.data, iris.target))
        self.assertAlmostEqual(grid_size, GridSearch.objects.get(uuid=gs._uuid).results.count(), delta=5)
        
    def test_ATGridSearchCV_with_dataset(self):
        examples, labels = _create_dataset()
        ds, _ = DataSet.objects.get_or_create(name='IRIS', 
                                              examples=SimpleUploadedFile(examples.name, examples.read()),
                                              labels=SimpleUploadedFile(labels.name, labels.read()))
        grid_size = 2*20*4
        gs = ATGridSearchCV(tree.DecisionTreeClassifier,{'criterion':['gini','entropy'],
                                                         'max_depth':range(1,21),
                                                         'max_features':['auto','log2','sqrt',None]},
                            client_kwargs={'address':LocalCluster()}, dataset=ds.pk,
                            webserver_url=self.live_server_url)
        wait(gs.fit())
        self.assertAlmostEqual(grid_size, GridSearch.objects.get(uuid=gs._uuid).results.count(), delta=5)
        
    def test_ATGridsSearchCV_without_dataset_and_fit(self):
        gs = ATGridSearchCV(tree.DecisionTreeClassifier,{'criterion':['gini','entropy'],
                                                         'max_depth':range(1,21),
                                                         'max_features':['auto','log2','sqrt',None]},
                            client_kwargs={'address':LocalCluster()},
                            webserver_url=self.live_server_url)
        with self.assertRaises(NoDatasetError):
            gs.fit()
    
class TestViews(LiveServerTestCase):
    def setUp(self):
        super(TestViews,self).setUp()
        GridSearch.objects.all().delete()
        DataSet.objects.all().delete()
        try:
            shutil.rmtree('media/datasets/IRIS/train/')
            shutil.rmtree('media/datasets/IRIS/test/')
        except FileNotFoundError:
            pass
    
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
        response = client.post(reverse('datasets'), data={'postMeta':'IRIS', 'items[]': [examples_file, label_file]})
        self.assertEqual(201, response.status_code)
        self.assertEqual(3, len(response.data))
        self.assertEqual(1, DataSet.objects.count())
    
    @unittest.skip
    def test_dataset_grids_results(self):
        client = DjangoClient()
        examples, labels = _create_dataset()
        ds, _ = DataSet.objects.get_or_create(name='IRIS', 
                                              examples=SimpleUploadedFile(examples.name, examples.read()),
                                              labels=SimpleUploadedFile(labels.name, labels.read()))
        grid_size = 2*20*4
        gs = ATGridSearchCV(tree.DecisionTreeClassifier,{'criterion':['gini','entropy'],
                                                         'max_depth':range(1,21),
                                                         'max_features':['auto','log2','sqrt',None]},
                            client_kwargs={'address':LocalCluster()}, dataset=ds.pk,
                            webserver_url=self.live_server_url)
        gs.fit()
        gs_1 = ATGridSearchCV(ensemble.RandomForestClassifier,{'criterion':['gini','entropy'],
                                                         'max_depth':range(1,21),
                                                         'max_features':['auto','log2','sqrt',None]},
                            client_kwargs={'address':LocalCluster()}, dataset=ds.pk,
                            webserver_url=self.live_server_url)
        gs_1.fit()
        response = client.get(reverse('dataset_grids', kwargs={'name': 'IRIS'}))
        
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
        
class TestAdminCommand(LiveServerTestCase):
    
    def setUp(self):
        super(TestAdminCommand,self).setUp()
        GridSearch.objects.all().delete()
        DataSet.objects.all().delete()
        try:
            shutil.rmtree('media/datasets/IRIS/train/')
            shutil.rmtree('media/datasets/IRIS/test/')
        except FileNotFoundError:
            pass
    
    def test_iris_tree(self):
        COMMAND = 'iris_experiments'
        call_command(COMMAND, 'Tree',url=self.live_server_url)
        self.assertEqual(1, GridSearch.objects.filter(classifier='DecisionTreeClassifier').count(), 
                         GridSearch.objects.filter(classifier='DecisionTreeClassifier'))
        gs_tree = GridSearch.objects.get(classifier='DecisionTreeClassifier')
        self.assertAlmostEqual(gs_tree.results.all().count(), 
                               2*5*(len(load_iris().data[0])-1),
                               msg=GridSearch.objects.filter(classifier='DecisionTreeClassifier'),
                               delta=5)
        
    def test_iris_forest(self):
        COMMAND = 'iris_experiments'
        call_command(COMMAND, 'Forest', url=self.live_server_url)
        self.assertEqual(1, GridSearch.objects.filter(classifier='RandomForestClassifier').count(), 
                         GridSearch.objects.filter(classifier='RandomForestClassifier'))
        gs_forest = GridSearch.objects.get(classifier='RandomForestClassifier')
        self.assertAlmostEqual(gs_forest.results.count(), 2*5*(len(load_iris().data[0])-1), delta=5)
        
    def test_iris_network(self):
        COMMAND = 'iris_experiments'
        call_command(COMMAND, 'Network', url=self.live_server_url)
        self.assertEqual(1, GridSearch.objects.filter(classifier='MLPClassifier').count()) 
        gs_network = GridSearch.objects.get(classifier='MLPClassifier')
        self.assertAlmostEqual(gs_network.results.count(), 81, delta=5)
