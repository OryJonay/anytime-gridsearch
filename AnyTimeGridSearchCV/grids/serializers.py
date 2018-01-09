'''
Created on May 15, 2017

@author: Ory Jonay
'''
from rest_framework import serializers
from AnyTimeGridSearchCV.grids.models import GridSearch, CVResult, DataSet

class GridSearchSerializer(serializers.ModelSerializer):
    
    class Meta:
        model = GridSearch
        fields = '__all__'
        
class CVResultSerializer(serializers.ModelSerializer):
    
    class Meta:
        model = CVResult
        fields = '__all__' 
        
class DatasetSerializer(serializers.ModelSerializer):
    
    class Meta:
        model = DataSet
        fields = '__all__'
        