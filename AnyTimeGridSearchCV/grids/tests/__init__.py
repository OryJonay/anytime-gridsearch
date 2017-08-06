from _io import BytesIO
import shutil

from django.test import LiveServerTestCase
import numpy
from sklearn.datasets.base import load_iris

from AnyTimeGridSearchCV.grids.models import GridSearch, DataSet


def _create_dataset():
    examples_file, label_file = BytesIO(), BytesIO()
    examples_file.name = 'examples.csv'
    label_file.name = 'labels.csv'
    iris = load_iris()
    numpy.savetxt(examples_file, iris.data, delimiter=',')
    numpy.savetxt(label_file, iris.target, delimiter=',')
    examples_file.seek(0), label_file.seek(0)
    return examples_file, label_file
    
class AbstractGridsTestCase(LiveServerTestCase):
    
    def setUp(self):
        super(AbstractGridsTestCase,self).setUp()
        GridSearch.objects.all().delete()
        DataSet.objects.all().delete()
        try:
            shutil.rmtree('media/datasets/IRIS/train/')
            shutil.rmtree('media/datasets/IRIS/test/')
        except FileNotFoundError:
            pass