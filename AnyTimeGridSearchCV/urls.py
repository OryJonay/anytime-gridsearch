"""AnyTimeGridSearchCV URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.contrib import admin

from AnyTimeGridSearchCV.grids.views import GridsListView, GridDetailView, \
    GridResultsList, GridResultTemplateView, DataSetsList, \
    DataSetGridsListView, EstimatorsListView, EstimatorDetailView


urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^grids/$', GridsListView.as_view(), name='grids_list'),
    url(r'^grids/(?P<uuid>.+)/$', GridDetailView.as_view(), name='grid_detail'),
    url(r'^grids/(?P<uuid>.+)/show$', GridResultTemplateView.as_view(), name='search_results_gui'),
    url(r'^grids/(?P<uuid>.+)/results$', GridResultsList.as_view(), name='grid_results'),
    url(r'^datasets/$', DataSetsList.as_view(), name='datasets'),
    url(r'^datasets/(?P<name>.+)/grids$', DataSetGridsListView.as_view(), name='dataset_grids'),
    url(r'^estimators/$', EstimatorsListView.as_view(), name="estimators_list"),
    url(r'^estimators/(?P<clf>.+)$', EstimatorDetailView.as_view(), name="estimator_detail"),
#     url(r'^', GridsTemplateView.as_view(), name='grids_index'),
]
