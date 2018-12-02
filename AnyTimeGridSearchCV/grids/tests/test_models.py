from django.core.files.uploadedfile import SimpleUploadedFile
import numpy
from sklearn import linear_model, tree
from sklearn.datasets.base import load_iris

from AnyTimeGridSearchCV.grids.models import DataSet, GridSearch, CVResult,\
    CVResultScore
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
        examples_file, label_file = _create_dataset()
        ds, _ = DataSet.objects.get_or_create(name='TEST',
                                              examples=SimpleUploadedFile(examples_file.name, examples_file.read()),
                                              labels=SimpleUploadedFile(label_file.name, label_file.read()))
        GridSearch.objects.get_or_create(classifier=reg.__class__.__name__, dataset=ds)

    def test_grid_search_result_model_creation(self):
        reg = linear_model.LinearRegression()
        _gs, _ = GridSearch.objects.get_or_create(classifier=reg.__class__.__name__)
        cv_result, _ = CVResult.objects.get_or_create(params=reg.get_params(), gridsearch=_gs)
        self.assertFalse(cv_result.scores.exists())

    def test_grid_seach_results_scores_model_creation(self):
        clf = tree.DecisionTreeClassifier()
        _gs, _ = GridSearch.objects.get_or_create(classifier=clf.__class__.__name__)
        cv_result, _ = CVResult.objects.get_or_create(params=clf.get_params(), gridsearch=_gs)
        CVResultScore.objects.create(score=0.5, scorer='accuracy',
                                                       train_scores=[0.5, 0.5], test_scores=[0.5, 0.5],
                                                       cv_result=cv_result)
        CVResultScore.objects.create(score=0.5, scorer='f1',
                                                       train_scores=[0.6, 0.7], test_scores=[0.4, 0.6],
                                                       cv_result=cv_result)
        self.assertEqual(2, cv_result.scores.all().count())
