from django.urls import path
from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^$', views.home, name="home"),
    path('translate/', views.machine_translation, name='machine_translation'),
    path('results/', views.view_results, name='results'),
]