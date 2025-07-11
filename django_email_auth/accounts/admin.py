from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import User


@admin.register(User)
class CustomUserAdmin(BaseUserAdmin):
    #model = User
    list_display = (
        "email",
        "username",
        'is_staff',
        "is_email_verified",
        "is_active",
        "date_joined",
    )
    list_filter = ("is_email_verified", "is_active", "is_staff", "date_joined")
    ordering = ("date_joined",)

    fieldsets = (
        (None, {'fields': ('email', 'username', 'password')}),
        ('Permissions', {'fields': ('is_staff', 'is_active', 'is_superuser', 'groups', 'user_permissions')}),
        ('Dates', {'fields': ('last_login', 'date_joined')}),
        ("Email Verification", {"fields": ("is_email_verified",)}),
    )
    
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('email', 'username', 'password1', 'password2', 'is_staff', 'is_active')}
        ),
    )

    search_fields = ("email", "username")