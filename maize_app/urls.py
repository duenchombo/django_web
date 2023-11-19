from django.urls import path
from .views import homePageView
from maize_app import views

urlpatterns = [
    path("", homePageView, name="home"),
    path("image_predict", views.image_predict, name="predict"),
]
