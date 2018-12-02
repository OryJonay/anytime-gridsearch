import json

import coreapi
import coreschema
from django.db.utils import IntegrityError
from django.shortcuts import get_object_or_404
from django.utils.datastructures import MultiValueDictKeyError
from numpydoc import docscrape
from rest_framework import status, schemas
from rest_framework.generics import ListAPIView, RetrieveAPIView, \
    ListCreateAPIView
from rest_framework.response import Response
from rest_framework.views import APIView

from AnyTimeGridSearchCV.grids.anytime_search import ESTIMATORS_DICT, \
    _convert_clf_param, ATGridSearchCV
from AnyTimeGridSearchCV.grids.models import GridSearch, CVResult, DataSet, \
    CVResultScore
from AnyTimeGridSearchCV.grids.serializers import GridSearchSerializer, \
    CVResultSerializer, DatasetSerializer


class EstimatorsListView(APIView):
    """
    Returns a list of all available scikit-learn classifiers.
    """

    def get(self, request, *args, **kwargs):
        return Response(list(ESTIMATORS_DICT.keys()), status=status.HTTP_200_OK)


class EstimatorDetailView(APIView):
    """
    Returns a detailed view of a scikit-learn classifier - all available arguments for the classifier.
    """

    schema = schemas.AutoSchema(manual_fields=[
        coreapi.Field(
            'clf',
            required=True,
            location='path',
            schema=coreschema.String(
                description='scikit-learn Estimator name'
            )
        ),
    ])

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


class GridsListView(ListCreateAPIView):
    """
    get:
    Returns a list of all available grid searches.

    post:
    Creates a new grid search.
    """

    queryset = GridSearch.objects.all()
    serializer_class = GridSearchSerializer

    def post(self, request, *args, **kwargs):
        return ListCreateAPIView.post(self, request, *args, **kwargs)


class GridDetailView(RetrieveAPIView):
    """
    Returns the specified grid (uuid, dataset name and scikit-learn classifier name).
    """

    queryset = GridSearch.objects.all()
    serializer_class = GridSearchSerializer
    lookup_field = 'uuid'


class GridResultsListSchema(schemas.AutoSchema):

    def get_manual_fields(self, path, method):
        manual_fields = schemas.AutoSchema.get_manual_fields(self, path, method)
        if method == 'GET':
            return manual_fields
        elif method == 'POST':
            return manual_fields + [coreapi.Field('cv_data', required=True, location='form',
                                                  schema=coreschema.Object(description='Cross validation result'))]


class GridResultsList(ListCreateAPIView):
    """
    get:
    Returns a list of all the results (CV classifications) for given grid.

    post:
    Creates a new result instance for specified grid.
    """

    queryset = CVResult.objects.all()
    serializer_class = CVResultSerializer

    schema = GridResultsListSchema(manual_fields=[
        coreapi.Field(
            'uuid',
            required=True,
            location='path',
            schema=coreschema.String(
                description='GridSearch UUID'
            )
        ),
    ])

    def get_queryset(self):
        _gs = get_object_or_404(GridSearch, uuid=self.kwargs['uuid'])
        return _gs.results.all()

    def post(self, request, *args, **kwargs):
        import numpy
        _gs = get_object_or_404(GridSearch, uuid=self.kwargs['uuid'])
        multimetric_scores = json.loads(request.data['cv_data'])
        scorers = set(map(lambda j: j.split('_')[-1],
                          filter(lambda i: i != 'fit_time' and i != 'score_time',
                                 multimetric_scores)))
        cv_result, _ = CVResult.objects.get_or_create(gridsearch=_gs,
                                                      params=json.loads(request.data['params']))
        cv_result.fit_time = multimetric_scores['fit_time']
        cv_result.score_time = multimetric_scores['score_time']
        cv_result.save()
        CVResultScore.objects.bulk_create([CVResultScore(scorer=scorer, train_scores=multimetric_scores['train_%s' % scorer],
                                                         test_scores=multimetric_scores['test_%s' % scorer],
                                                         score=round(numpy.array(multimetric_scores[
                                                                     'test_%s' % scorer]).mean(), 6),
                                                         cv_result=cv_result) for scorer in scorers])
        return Response(CVResultSerializer(cv_result).data, status=status.HTTP_201_CREATED)


class DataSetsList(ListCreateAPIView):
    """
    get:
    Returns a list of all the existing Datasets.

    post:
    Creates a new Dataset instance.
    """

    queryset = DataSet.objects.all()
    serializer_class = DatasetSerializer

    def post(self, request, *args, **kwargs):
        import numpy
        try:
            name = request.data['name']
        except MultiValueDictKeyError:
            return Response('Missing dataset name', status=status.HTTP_400_BAD_REQUEST)
        if not name:
            return Response('Missing dataset name', status=status.HTTP_400_BAD_REQUEST)
        try:
            examples, labels = request.FILES['examples'], request.FILES['labels']
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
    """
    Returns all grid searches on the given Dataset.
    """

    queryset = GridSearch.objects.all()
    serializer_class = GridSearchSerializer

    schema = schemas.AutoSchema(manual_fields=[
        coreapi.Field(
            'name',
            required=True,
            location='path',
            schema=coreschema.String(
                description='Dataset name'
            )
        ),
    ])

    def get_queryset(self):
        _ds = get_object_or_404(DataSet, name=self.kwargs['name'])
        return _ds.grid_searches.all()


class ATGridSearchCreateView(APIView):
    """
    Creates a new ATGridSearch instance (with the grid specified in the request) and starts it.
    """

    schema = schemas.AutoSchema(manual_fields=[
        coreapi.Field(
            'dataset',
            required=True,
            location='form',
            schema=coreschema.String(description='Dataset name')
        ),
        coreapi.Field(
            'clf',
            required=True,
            location='form',
            schema=coreschema.String(description='scikit-learn estimator name')
        ),
        coreapi.Field(
            'args',
            required=True,
            location='form',
            schema=coreschema.Object(description='Grid to search'),
        ),

    ])

    def post(self, request, *args, **kwargs):
        try:
            ds = DataSet.objects.get(name=request.data['dataset'])
        except DataSet.DoesNotExist:
            return Response('No DataSet named {}'.format(request.data['dataset']), status=status.HTTP_400_BAD_REQUEST)
        try:
            classifier = ESTIMATORS_DICT[request.data['clf']]
        except KeyError:
            return Response('No sklearn classifier named {}'.format(request.data['clf']), status=status.HTTP_400_BAD_REQUEST)
        clf_params = {k: _convert_clf_param(v) for k, v in request.data['args'].items()}
        gs = ATGridSearchCV(classifier(), clf_params, dataset=ds.pk)
        gs.fit()
        return Response(gs._uuid, status=status.HTTP_201_CREATED)
