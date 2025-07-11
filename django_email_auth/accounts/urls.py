from django.urls import path
from .views import CustomLoginView, CustomLogoutView, SignUpView, EmailVerificationView
from django.contrib.auth.views import PasswordResetView, PasswordResetConfirmView

urlpatterns = [
    path("signup/", SignUpView.as_view(), name="signup"),
    path("login/", CustomLoginView.as_view(), name="login"),
    path("logout/", CustomLogoutView.as_view(), name="logout"),
    path("verify-email/<uidb64>/<token>/", EmailVerificationView.as_view(), name="verify_email"),

    # (опционально — сброс пароля, если понадобится)
    path("password-reset/", PasswordResetView.as_view(), name="password_reset"),
    path("reset/<uidb64>/<token>/", PasswordResetConfirmView.as_view(), name="password_reset_confirm"),
]
