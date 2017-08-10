from django.core.files.uploadedfile import SimpleUploadedFile
from django.db.models.expressions import RawSQL
import numpy
from sklearn import linear_model
from sklearn.datasets.base import load_iris

from AnyTimeGridSearchCV.grids.models import DataSet, GridSearch, CVResult
from AnyTimeGridSearchCV.grids.tests import AbstractGridsTestCase, \
    _create_dataset


class TestModels(AbstractGridsTestCase):
    
    def test_dataset_model_single_file(self):
        examples_file, label_file = _create_dataset()
        ds, _ = DataSet.objects.get_or_create(name='TEST', 
                                              examples=SimpleUploadedFile(examples_file.name, examples_file.read()),
                                              labels=SimpleUploadedFile(label_file.name, label_file.read()))
        self.assertEqual('datasets/TEST/examples.csv', ds.examples.name)
        self.assertEqual('datasets/TEST/labels.csv', ds.labels.name)
        loaded_train = numpy.genfromtxt(ds.examples, delimiter=',')
        loaded_labels = numpy.genfromtxt(ds.labels, delimiter=',')
        iris = load_iris()
        self.assertTrue(numpy.array_equal(loaded_train, iris.data))
        self.assertTrue(numpy.array_equal(loaded_labels, iris.target))
            
    def test_grid_search_model_creation(self):
        reg = linear_model.LinearRegression()
        examples_file , label_file = _create_dataset()
        ds, _ = DataSet.objects.get_or_create(name='TEST', 
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