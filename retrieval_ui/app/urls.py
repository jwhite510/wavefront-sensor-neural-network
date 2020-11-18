from app import views
from django.conf.urls import url

app_name='app'

urlpatterns=[
        url(r'^$', views.home, name='home'),
        ]
