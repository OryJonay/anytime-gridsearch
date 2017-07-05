import os
import uuid

from django.contrib.postgres import fields
from django.db import models
from django.db.models.expressions import RawSQL


# Create your models here.
class GridSearch(models.Model):
    
    uuid = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    classifier = models.CharField(max_length=128, blank=False, null=False)
    dataset = models.ForeignKey('DataSet', null=True, blank=True, related_name='grid_searches')
    
    def order_results_by(self,inner_field,limit=100):
        top_n = self.results.filter(**{'params__%s__isnull' % inner_field:False}).order_by('-score')[:limit]
        return self.results.filter(id__in=top_n).order_by(RawSQL("params->>%s", (inner_field,)))
    
class CVResult(models.Model):
    
    score = models.FloatField(blank=False, null=False)
    params = fields.JSONField(default=dict)
    gridsearch = models.ForeignKey(GridSearch, related_name='results')
    cross_validation_scores = fields.ArrayField(models.FloatField(default=0.), default=list)

def _path_to_upload_train(instance, filename):
    return os.path.join('datasets', instance.name, 'train', filename)

def _path_to_upload_test(instance, filename):
    return os.path.join('datasets', instance.name, 'test', filename)
    
class DataSet(models.Model):
    
    name = models.CharField(max_length=256, primary_key=True)
    examples = models.FileField(upload_to=_path_to_upload_train)
    labels = models.FileField(upload_to=_path_to_upload_test)