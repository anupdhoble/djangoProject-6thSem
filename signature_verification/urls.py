from django.http import HttpResponse
from django.urls import path
from . import views

urlpatterns = [
    path('', lambda request: HttpResponse("Hello, HTML!"), name='home'),
    path('classify/', views.classify_signature, name='classify_signature'),  # Add this line for the new view
]
