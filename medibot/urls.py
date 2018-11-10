from django.conf.urls import url
from . import views

app_name = 'medibot'
urlpatterns = [
    # 챗봇
    url(r'^$', views.index, name='index'),
    url(r'^ajaxpost$', views.ajaxpost, name='ajaxpost'),
    # 추가 기능
    url(r'^autocomplete', views.autocomplete, name='autocomplete'),
    url(r'^physical', views.physical, name='physical'),
    url(r'^diet', views.diet, name='diet'),
    url(r'^hist', views.hist, name='hist'),
    # url(r'^bmi$', views.bmi, name='bmi'),
    # url(r'^whr$', views.whr, name='whr'),
    # url(r'^energy$', views.energy, name='energy'),
    # 테스트용
    url(r'^test$', views.test, name='test'),
]



