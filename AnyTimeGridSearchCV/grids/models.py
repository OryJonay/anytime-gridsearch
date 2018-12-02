import os
import uuid

from django.contrib.postgres import fields
from django_cleanup.signals import cleanup_pre_delete
from django.db import models
import shutil


class GridSearch(models.Model):

    uuid = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False,
                            help_text='Unique identifier for the GridSearch')
    classifier = models.CharField(max_length=128, blank=False, null=False,
                                  help_text='scikit-learn estimator name')
    dataset = models.ForeignKey('DataSet', null=True, blank=True, related_name='grid_searches', on_delete=models.CASCADE,
                                help_text='The dataset on which the GridSearch will run')


class CVResult(models.Model):

    params = fields.JSONField(default=dict,
                              help_text='Dictionary of hyperparameters for the estimator')
    gridsearch = models.ForeignKey(GridSearch, related_name='results', on_delete=models.CASCADE,
                                   help_text='The GridSearch instance that this result belongs to')
    fit_time = fields.ArrayField(models.FloatField(default=0.), default=list,
                                 help_text='Estimator fit time on cross validation')
    score_time = fields.ArrayField(models.FloatField(default=0.), default=list,
                                   help_text='Estimator score time on cross validation')
    errors = models.TextField(null=True, blank=True,
                              help_text='Various errors that happened in the run')


class CVResultScore(models.Model):

    scorer = models.CharField(max_length=256,
                              help_text='Score function name')
    cv_result = models.ForeignKey(CVResult, related_name='scores', on_delete=models.CASCADE,
                                  help_text='The CVResult instance that the score belongs to')
    train_scores = fields.ArrayField(models.FloatField(default=0.), default=list,
                                     help_text='Cross validation train scores')
    test_scores = fields.ArrayField(models.FloatField(default=0.), default=list,
                                    help_text='Cross validation test scores')
    score = models.FloatField(default=0.,
                              help_text='Mean score for cross validation')


def _path_to_upload_train(instance, filename):
    return os.path.join('datasets', instance.name, filename)


def _path_to_upload_test(instance, filename):
    return os.path.join('datasets', instance.name, filename)


class DataSet(models.Model):

    name = models.CharField(max_length=256, primary_key=True,
                            help_text='Dataset name')
    examples = models.FileField(upload_to=_path_to_upload_train,
                                help_text='Examples file')
    labels = models.FileField(upload_to=_path_to_upload_test,
                              help_text='Labels file')


def dataset_cleanup(**kwargs):
    if len(os.listdir(os.path.dirname(kwargs['file'].path))) == 1:
        shutil.rmtree(os.path.dirname(kwargs['file'].path), ignore_errors=True)


cleanup_pre_delete.connect(dataset_cleanup)
