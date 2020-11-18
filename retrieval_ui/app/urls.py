from app import views
from django.conf.urls import url

app_name='app'

urlpatterns=[
        url(r'^$', views.home, name='home'),
        url(r'^retrieve',views.retrieve,name='retrieve')
        ]
