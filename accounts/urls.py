from django.conf.urls import url
from . import views
app_name = 'accounts'
urlpatterns = [
    # 회원인증
    url(r'^signup', views.signup, name='join'),
    url(r'^login$', views.signin, name='login'),
    url(r'^logout', views.signout, name='logout'),
    url(r'^search', views.search, name='search'),
    url('activate/<str:uidb63>/<str:token>', views.activate, name='activate')
]
