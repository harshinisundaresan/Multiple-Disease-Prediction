from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',views.home,name='home'),
    path('model1',views.model1,name='model1'),
    path('result1',views.result1,name='result1'),
    path('model2',views.model2,name='model2'),
    path('result2',views.result2,name='result2'),
    path('model3',views.model3,name='model3'),
    path('result3',views.result3,name='result3')

]
