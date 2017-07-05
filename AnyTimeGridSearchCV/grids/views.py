from django.shortcuts import get_object_or_404
from django.views.generic.list import ListView
from rest_framework.generics import ListAPIView, RetrieveAPIView, \
    ListCreateAPIView

from AnyTimeGridSearchCV.grids.models import GridSearch, CVResult, DataSet
from AnyTimeGridSearchCV.grids.serializers import GridSearchSerializer, \
    CVResultSerializer, DatasetSerializer


class GridsTemplateView(ListView):
    context_object_name = 'grids'
    queryset = GridSearch.objects.all()
    template_name = 'grids_index.html'
    

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
        return super(DataSetsList, self).post(request, *args, **kwargs)
    
class DataSetGridsListView(ListAPIView):
    
    queryset = GridSearch.objects.all()
    serializer_class = GridSearchSerializer
    
    def get_queryset(self):
        _ds = get_object_or_404(DataSet, name=self.kwargs['name'])
        return _ds.grid_searches.all()