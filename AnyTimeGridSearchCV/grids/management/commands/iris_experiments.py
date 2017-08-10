'''
Created on May 16, 2017

@author: Ory Jonay
'''
from io import BytesIO

from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management.base import BaseCommand
import numpy
from sklearn.datasets.base import load_iris, load_breast_cancer, load_diabetes
import sklearn.ensemble
import sklearn.neural_network
import sklearn.tree

from AnyTimeGridSearchCV.grids.anytime_search import ATGridSearchCV
from AnyTimeGridSearchCV.grids.models import DataSet
import distributed

def _create_dataset(dataset):
        examples_file, label_file = BytesIO(), BytesIO()
        examples_file.name = 'examples.csv'
        label_file.name = 'labels.csv'
        numpy.savetxt(examples_file, dataset.data, delimiter=',')
        numpy.savetxt(label_file, dataset.target, delimiter=',')
        examples_file.seek(0), label_file.seek(0)
        return examples_file, label_file

class Command(BaseCommand):
    
    help = 'Run different experiments on the Iris dataset to populate the database'
    
    CLASSIFIERS = {'Tree':sklearn.tree.DecisionTreeClassifier,
                   'Forest':sklearn.ensemble.RandomForestClassifier,
                   'Network':sklearn.neural_network.MLPClassifier}
    
    DATASETS = {'IRIS': load_iris,
                'TEST': load_iris,
                'BREAST_CANCER': load_breast_cancer,
                'DIABETES': load_diabetes}
    
    def add_arguments(self, parser):
        parser.add_argument('classifier',choices=list(self.CLASSIFIERS.keys()),
                            help='Classifier to run - DecisionTree / RandomForest / Multi-layer Perceptron')
        parser.add_argument('dataset', choices=list(self.DATASETS.keys()), nargs='?', default= 'IRIS',
                            help='Dataset to use')
        parser.add_argument('--url', nargs='?', default= 'http://127.0.0.1:8000',
                            help='URL of webserver to store results')
        
        
    def handle(self, *args, **options):
        dataset = self.DATASETS[options['dataset']]()
        example_f, labels_f = _create_dataset(dataset)
        client_kwargs = {'address': distributed.LocalCluster()}
        try:
            ds = DataSet.objects.get(name=options['dataset'])
        except DataSet.DoesNotExist:
            ds = DataSet.objects.create(name=options['dataset'], 
                                      examples=SimpleUploadedFile(example_f.name, example_f.read()),
                                      labels=SimpleUploadedFile(labels_f.name, labels_f.read()))
                                          
        if options['classifier'] == 'Tree':
            gs_tree = ATGridSearchCV(sklearn.tree.DecisionTreeClassifier,{'criterion':['gini','entropy'],
                                                         'max_depth':range(1,6),
                                                         'max_features':range(1,len(dataset.data[0]))},
                                     dataset=ds.name,
                                     webserver_url=options['url'],
                                     client_kwargs=client_kwargs)
            futures = gs_tree.fit(dataset.data, dataset.target)
            distributed.wait(futures)
        elif options['classifier'] == 'Forest':
            gs_forest = ATGridSearchCV(sklearn.ensemble.RandomForestClassifier,{'criterion':['gini','entropy'],
                                                         'max_depth':range(1,6),
                                                         'max_features':range(1,len(dataset.data[0]))},
                                       dataset=ds.name,
                                       webserver_url=options['url'],
                                       client_kwargs=client_kwargs)
            distributed.wait(gs_forest.fit(dataset.data,dataset.target))
        else:
            gs_network = ATGridSearchCV(sklearn.neural_network.MLPClassifier,{'solver' : ['lbfgs','sgd','adam'],
                                                         'learning_rate':['constant','invscaling','adaptive'],
                                                         'max_iter':range(200,2000,200)},
                                        dataset=ds.name,
                                        webserver_url=options['url'],
                                        client_kwargs=client_kwargs)
            distributed.wait(gs_network.fit(dataset.data,dataset.target))
