
from django.urls import path
from . import views

urlpatterns = [
    path('h1', views.home1, name='home1'),
    path('', views.predict_fire_alarm, name='predict_fire_alarm'),

]



