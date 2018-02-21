'''
Created on May 15, 2017

@author: Ory Jonay
'''
from rest_framework import serializers
from AnyTimeGridSearchCV.grids.models import GridSearch, CVResult, DataSet, CVResultScore

class GridSearchSerializer(serializers.ModelSerializer):
    
    class Meta:
        model = GridSearch
        fields = '__all__'
        
class CVResultScoreSeriazlier(serializers.ModelSerializer):
    
    class Meta:
        model = CVResultScore
        fields = ('scorer', 'score')
        
class CVResultSerializer(serializers.ModelSerializer):
    
    scores = CVResultScoreSeriazlier(many=True, read_only=True)
    
    class Meta:
        model = CVResult
        fields = ('params', 'gridsearch', 'errors', 'scores', 'id', 'fit_time') 
        
class DatasetSerializer(serializers.ModelSerializer):
    
    class Meta:
        model = DataSet
        fields = '__all__'
        