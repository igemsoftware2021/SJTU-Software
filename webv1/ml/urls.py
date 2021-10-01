from django.urls import path, re_path
from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'arrays/$', views.arraysGenericAPIView.as_view()),
    url(r'^array/(?P<pk>\d+)/$', views.arrayGenericAPIView.as_view()),

    url(r'svms/$', views.svmsGenericAPIView.as_view()),
    url(r'^svm/(?P<pk>\d+)/$', views.svmGenericAPIView.as_view()),
]
