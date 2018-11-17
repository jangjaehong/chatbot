from django.db import models
from django.contrib.auth.models import BaseUserManager, AbstractBaseUser, PermissionsMixin

class UserManager(BaseUserManager):
    def create_user(self, username, email, mobile, gender, birth, password=None):
        if not email:
            raise ValueError("이메일 입력해주세요!")

        user = self.model(
            email=self.normalize_email(email),
            username=username,
            mobile=mobile,
            gender=gender,
            birth=birth
        )
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, username, email, mobile, password=None):
        user = self.create_user(
            email=email,
            username=username,
            mobile=mobile,
            password=password,
        )
        user.is_admin = True
        user.is_active = True
        user.save(using=self._db)
        return user


class User(AbstractBaseUser, PermissionsMixin):
    email = models.EmailField(max_length=255, default="", unique=True)
    mobile = models.CharField(max_length=13, default="", unique=True)
    username = models.CharField(max_length=30, default="")
    gender = models.IntegerField(default=0)
    birth = models.CharField(max_length=8, default="19000101")
    create_at = models.DateTimeField(auto_now_add=True)
    update_at = models.DateTimeField(auto_now=True)
    is_admin = models.BooleanField(default=False)
    is_active = models.BooleanField(default=False)

    objects = UserManager()

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username', 'mobile']

    def get_full_name(self):
        # The user is identified by their email address
        return self.email

    def get_short_name(self):
        # The user is identified by their email address
        return self.email

    def __str__(self):  # __unicode__ on Python 2
        return self.email

    def has_perm(self, perm, obj=None):
        "Does the user have a specific permission?"
        # Simplest possible answer: Yes, always
        return True

    def has_module_perms(self, app_label):
        "Does the user have permissions to view the app `app_label`?"
        # Simplest possible answer: Yes, always
        return True

    @property
    def is_staff(self):
        "Is the user a member of staff?"
        # Simplest possible answer: All admins are staff
        return self.is_admin