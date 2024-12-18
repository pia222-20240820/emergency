from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # 메인 페이지 (병원 검색 및 질문 처리)
    path('reset-chat', views.reset_chat, name='reset_chat'),
]
