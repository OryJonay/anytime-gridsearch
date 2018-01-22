from _io import BytesIO
import shutil

from django.test import LiveServerTestCase
import numpy
from sklearn.datasets.base import load_iris

from django.db.utils import IntegrityError

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
        from AnyTimeGridSearchCV.grids.models import GridSearch, DataSet, CVResult, CVResultScore

        super(AbstractGridsTestCase,self).setUp()
        CVResultScore.objects.all().delete()
        CVResult.objects.all().delete()
        try:
            GridSearch.objects.all().delete()
        except IntegrityError: # pragma: no cover
            CVResult.objects.all().delete() 
            GridSearch.objects.all().delete() 
        DataSet.objects.all().delete()
        try:
            shutil.rmtree('media/datasets/TEST/')
        except FileNotFoundError:
            pass
        
    def tearDown(self):
        from AnyTimeGridSearchCV.grids.models import GridSearch, DataSet, CVResult, CVResultScore

        super(AbstractGridsTestCase,self).tearDown()
        CVResultScore.objects.all().delete()
        CVResult.objects.all().delete()
        try:
            GridSearch.objects.all().delete()
        except IntegrityError: # pragma: no cover
            pass 
        DataSet.objects.all().delete()
        try:
            shutil.rmtree('media/datasets/TEST/')
        except FileNotFoundError:
            pass