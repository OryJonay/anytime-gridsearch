from django.core.management import call_command
from sklearn.datasets.base import load_iris

from AnyTimeGridSearchCV.grids.models import GridSearch
from AnyTimeGridSearchCV.grids.tests import AbstractGridsTestCase


class TestAdminCommand(AbstractGridsTestCase):

    def test_iris_tree(self):
        COMMAND = 'iris_experiments'
        call_command(COMMAND, 'Tree', 'TEST', url=self.live_server_url)
        self.assertEqual(1, GridSearch.objects.filter(classifier='DecisionTreeClassifier').count(),
                         GridSearch.objects.filter(classifier='DecisionTreeClassifier'))
        gs_tree = GridSearch.objects.get(classifier='DecisionTreeClassifier')
        self.assertAlmostEqual(gs_tree.results.all().count(),
                               2 * 5 * (len(load_iris().data[0]) - 1),
                               msg=GridSearch.objects.filter(classifier='DecisionTreeClassifier'),
                               delta=5)

    def test_iris_forest(self):
        COMMAND = 'iris_experiments'
        call_command(COMMAND, 'Forest', 'TEST', url=self.live_server_url)
        self.assertEqual(1, GridSearch.objects.filter(classifier='RandomForestClassifier').count(),
                         GridSearch.objects.filter(classifier='RandomForestClassifier'))
        gs_forest = GridSearch.objects.get(classifier='RandomForestClassifier')
        self.assertAlmostEqual(gs_forest.results.count(), 2 * 5 * (len(load_iris().data[0]) - 1), delta=5)

    def test_iris_network(self):
        COMMAND = 'iris_experiments'
        call_command(COMMAND, 'Network', 'TEST', url=self.live_server_url)
        self.assertEqual(1, GridSearch.objects.filter(classifier='MLPClassifier').count())
        gs_network = GridSearch.objects.get(classifier='MLPClassifier')
        self.assertAlmostEqual(gs_network.results.count(), 81, delta=5)
