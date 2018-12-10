from django.conf.urls import url
from . import views

app_name = 'medibot'
urlpatterns = [
    # 챗봇
    url(r'^$', views.index, name='index'),
    url(r'^ajaxpost$', views.ajaxpost, name='ajaxpost'),
    # 추가 기능
    url(r'^autocomplete', views.autocomplete, name='autocomplete'),
    url(r'^physical_update', views.physical_update, name='physical_update'),
    url(r'^day_measure', views.day_measure, name='day_measure'),
    url(r'^diet', views.diet, name='diet'),
    url(r'^hist', views.hist, name='hist'),
    # 테스트용
    #url(r'^test$', views.test, name='test'),
]



