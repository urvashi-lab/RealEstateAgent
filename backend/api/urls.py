from django.urls import path
from .views import analyze
from .views import analyze, download_pdf 

urlpatterns = [
    path("analyze/", analyze),
    path("download-pdf/", download_pdf, name='download_pdf'),
]
