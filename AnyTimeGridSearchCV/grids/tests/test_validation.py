from distributed.client import wait, Client
from distributed.deploy.local import LocalCluster
from django.core.files.uploadedfile import SimpleUploadedFile
from sklearn import tree
from sklearn.datasets.base import load_iris
from sklearn.model_selection._validation import cross_val_score

from AnyTimeGridSearchCV.grids.anytime_search import ATGridSearchCV, \
    NoDatasetError
from AnyTimeGridSearchCV.grids.models import GridSearch, CVResult, DataSet
from AnyTimeGridSearchCV.grids.tests import AbstractGridsTestCase, \
    _create_dataset


class TestValidation(AbstractGridsTestCase):
    
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
        gs = ATGridSearchCV(tree.DecisionTreeClassifier(),{'criterion':['gini','entropy'],
                                                         'max_depth':range(1,21),
                                                         'max_features':['auto','log2','sqrt',None]},
                            webserver_url=self.live_server_url)
        wait(gs.fit(iris.data, iris.target))
        self.assertAlmostEqual(grid_size, GridSearch.objects.get(uuid=gs._uuid).results.count(), delta=5)
        
    def test_ATGridSearchCV_with_dataset(self):
        examples, labels = _create_dataset()
        ds, _ = DataSet.objects.get_or_create(name='TEST', 
                                              examples=SimpleUploadedFile(examples.name, examples.read()),
                                              labels=SimpleUploadedFile(labels.name, labels.read()))
        grid_size = 2*20*4
        gs = ATGridSearchCV(tree.DecisionTreeClassifier(),{'criterion':['gini','entropy'],
                                                         'max_depth':range(1,21),
                                                         'max_features':['auto','log2','sqrt',None]},
                            dataset=ds.pk,
                            webserver_url=self.live_server_url)
        wait(gs.fit())
        self.assertAlmostEqual(grid_size, GridSearch.objects.get(uuid=gs._uuid).results.count(), delta=5)
        
    def test_ATGridsSearchCV_without_dataset_and_fit(self):
        gs = ATGridSearchCV(tree.DecisionTreeClassifier(),{'criterion':['gini','entropy'],
                                                         'max_depth':range(1,21),
                                                         'max_features':['auto','log2','sqrt',None]},
                            webserver_url=self.live_server_url)
        with self.assertRaises(NoDatasetError):
            gs.fit()