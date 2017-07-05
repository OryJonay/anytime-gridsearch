'''
Created on May 15, 2017

@author: Ory Jonay
'''
from django.db.models import Max
from rest_framework import serializers
from AnyTimeGridSearchCV.grids.models import GridSearch, CVResult, DataSet

class GridSearchSerializer(serializers.ModelSerializer):
    
    best_score = serializers.SerializerMethodField()
    
    class Meta:
        model = GridSearch
        fields = ('uuid', 'classifier','best_score')
        
    def get_best_score(self,obj):
        return obj.results.aggregate(max_score=Max('score'))['max_score']
        
class CVResultSerializer(serializers.ModelSerializer):
    
#     gridsearch = serializers.PrimaryKeyRelatedField(read_only=False, queryset=GridSearch.objects.all())
    
    class Meta:
        model = CVResult
        fields = ('score', 'params', 'gridsearch', 'cross_validation_scores')
        
class DatasetSerializer(serializers.ModelSerializer):
    
    class Meta:
        model = DataSet
        fields = '__all__'
        