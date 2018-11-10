from django.conf.urls import url
from .views import *
app_name = 'accounts'
urlpatterns = [
    # 회원인증
    url(r'^join$', signup, name='join'),
    url(r'^login$', signin, name='login'),
    url(r'^logout', signin, name='logout'),
]
