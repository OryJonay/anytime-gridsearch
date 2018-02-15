import os
import uuid

from django.contrib.postgres import fields
from django.db import models

class GridSearch(models.Model):
    
    uuid = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    classifier = models.CharField(max_length=128, blank=False, null=False)
    dataset = models.ForeignKey('DataSet', null=True, blank=True, related_name='grid_searches', on_delete=models.CASCADE)
    
class CVResult(models.Model):
    
    params = fields.JSONField(default=dict)
    gridsearch = models.ForeignKey(GridSearch, related_name='results', on_delete=models.CASCADE)
    fit_time = fields.ArrayField(models.FloatField(default=0.), default=list)
    score_time = fields.ArrayField(models.FloatField(default=0.), default=list)
    errors = models.TextField(null=True, blank=True)
    
class CVResultScore(models.Model):
    
    scorer = models.CharField(max_length=256)
    cv_result = models.ForeignKey(CVResult, related_name='scores', on_delete=models.CASCADE)
    train_scores = fields.ArrayField(models.FloatField(default=0.), default=list)
    test_scores = fields.ArrayField(models.FloatField(default=0.), default=list)
    score = models.FloatField(default=0.)

def _path_to_upload_train(instance, filename):
    return os.path.join('datasets', instance.name, filename)

def _path_to_upload_test(instance, filename):
    return os.path.join('datasets', instance.name, filename)
    
class DataSet(models.Model):
    
    name = models.CharField(max_length=256, primary_key=True)
    examples = models.FileField(upload_to=_path_to_upload_train)
    labels = models.FileField(upload_to=_path_to_upload_test)
    