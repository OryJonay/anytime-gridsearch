from django.contrib.gis.shortcuts import numpy
from django.shortcuts import get_object_or_404
from django.views.generic.list import ListView
from rest_framework import status
from rest_framework.generics import ListAPIView, RetrieveAPIView, \
    ListCreateAPIView
from rest_framework.response import Response

from AnyTimeGridSearchCV.grids.models import GridSearch, CVResult, DataSet
from AnyTimeGridSearchCV.grids.serializers import GridSearchSerializer, \
    CVResultSerializer, DatasetSerializer
from AnyTimeGridSearchCV.grids.anytime_search import ESTIMATORS_DICT
from django.db.utils import IntegrityError
from django.utils.datastructures import MultiValueDictKeyError
from rest_framework.views import APIView
from numpydoc import docscrape



class EstimatorsListView(APIView):
    
    def get(self, request, *args, **kwargs):
        return Response(list(ESTIMATORS_DICT.keys()), status=status.HTTP_200_OK)

class EstimatorDetailView(APIView):
    
    def get(self, request, *args, **kwargs):
        try:
            clf = ESTIMATORS_DICT[kwargs.get('clf', 
                                             'Not a valid scikit-learn estimator name')]
        except KeyError:
            return Response({'name': '', 'type': '', 'desc': ''}, 
                        status=status.HTTP_200_OK)
        return Response([{'name': arg_name, 'type': arg_type, 'desc': arg_desc} 
                         for arg_name, arg_type, arg_desc in docscrape.ClassDoc(clf)['Parameters']], 
                        status=status.HTTP_200_OK)

class GridResultTemplateView(ListView):
    context_object_name = 'results'
    queryset = CVResult.objects.all()
    template_name = 'search_results.html'

    def get_queryset(self):
        _gs = get_object_or_404(GridSearch,uuid=self.kwargs['uuid'])
        return _gs.results.order_by('-score')
    
class GridsListView(ListCreateAPIView):
    
    queryset = GridSearch.objects.all()
    serializer_class = GridSearchSerializer
    
    def post(self, request, *args, **kwargs):
        return ListCreateAPIView.post(self, request, *args, **kwargs)
    
class GridDetailView(RetrieveAPIView):
    
    queryset = GridSearch.objects.all()
    serializer_class = GridSearchSerializer
    lookup_field = 'uuid'

class GridResultsList(ListCreateAPIView):
    
    queryset = CVResult.objects.all()
    serializer_class = CVResultSerializer
    
    def get_queryset(self):
        _gs = get_object_or_404(GridSearch, uuid=self.kwargs['uuid'])
        return _gs.results.order_by('-score')
    
class DataSetsList(ListCreateAPIView):
    
    queryset = DataSet.objects.all()
    serializer_class = DatasetSerializer
    
    def post(self, request, *args, **kwargs):
        try:
            name = request.data['dataset']
        except MultiValueDictKeyError:
            return Response('Missing dataset name', status=status.HTTP_400_BAD_REQUEST)
        if not name:
            return Response('Missing dataset name', status=status.HTTP_400_BAD_REQUEST)
        try:
            if len(request.FILES) > 2:
                return Response('Too many files', status=status.HTTP_400_BAD_REQUEST)
            examples, labels = request.FILES['file[0]'], request.FILES['file[1]'] 
        except MultiValueDictKeyError:
            return Response('Missing dataset files', status=status.HTTP_400_BAD_REQUEST)
        if examples.name != 'examples.csv':
            return Response('Bad name of examples file', status=status.HTTP_400_BAD_REQUEST)
        if labels.name != 'labels.csv':
            return Response('Bad name of labels file', status=status.HTTP_400_BAD_REQUEST)
        if len(numpy.genfromtxt(examples, delimiter=',')) != len(numpy.genfromtxt(labels, delimiter=',')):
            return Response('Examples and labels are not the same length', status=status.HTTP_400_BAD_REQUEST)
        try:
            return Response(DatasetSerializer(DataSet.objects.create(name=name, 
                                                                     examples=examples, 
                                                                     labels=labels)).data, 
                            status=status.HTTP_201_CREATED)
        except IntegrityError:
            return Response('Name already exists', status=status.HTTP_400_BAD_REQUEST)
    
class DataSetGridsListView(ListAPIView):
    
    queryset = GridSearch.objects.all()
    serializer_class = GridSearchSerializer
    
    def get_queryset(self):
        _ds = get_object_or_404(DataSet, name=self.kwargs['name'])
        return _ds.grid_searches.all()