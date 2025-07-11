from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate
from django.contrib.auth.views import LoginView, LogoutView
from django.views.generic import CreateView
from django.urls import reverse_lazy
from django.contrib import messages
from django.core.mail import send_mail
from django.template.loader import render_to_string
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils.encoding import force_bytes, force_str
from django.contrib.auth.tokens import default_token_generator
from django.views import View
from django.http import HttpResponse
from django.conf import settings

from .models import User
from .forms import CustomUserCreationForm, CustomAuthenticationForm


class CustomLoginView(LoginView):
    form_class = CustomAuthenticationForm
    template_name = "accounts/login.html"
    redirect_authenticated_user = True


class CustomLogoutView(LogoutView):
    next_page = "home"


class SignUpView(CreateView):
    form_class = CustomUserCreationForm
    template_name = "accounts/signup.html"
    success_url = reverse_lazy("login")

    def form_valid(self, form):
        response = super().form_valid(form)
        user = self.object
        user.is_active = False  # Деактивируем пользователя до подтверждения email
        user.save()

        # Отправляем email для подтверждения
        self.send_verification_email(user)

        messages.success(
            self.request,
            "Регистрация прошла успешно! Проверьте свою почту для подтверждения аккаунта.",
        )
        return response

    def send_verification_email(self, user):
        subject = "Подтвердите свою почту"
        token = default_token_generator.make_token(user)
        uid = urlsafe_base64_encode(force_bytes(user.pk))

        message = render_to_string(
            "accounts/verification_email.html",
            {
                "user": user,
                "domain": self.request.get_host(),
                "uid": uid,
                "token": token,
                "protocol": "https" if self.request.is_secure() else "http",
            },
        )

        send_mail(
            subject,
            "",
            settings.DEFAULT_FROM_EMAIL,
            [user.email],
            html_message=message,
            fail_silently=False,
        )


class EmailVerificationView(View):
    def get(self, request, uidb64, token):
        try:
            uid = force_str(urlsafe_base64_decode(uidb64))
            user = User.objects.get(pk=uid)
        except (TypeError, ValueError, OverflowError, User.DoesNotExist):
            user = None

        if user is not None and default_token_generator.check_token(user, token):
            user.is_active = True
            user.is_email_verified = True
            user.save()
            login(request, user)
            messages.success(request, "Ваша почта успешно подтверждена!")
            return redirect("home")
        else:
            messages.error(request, "Ссылка для подтверждения недействительна!")
            return redirect("login")
