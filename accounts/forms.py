from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm, UserChangeForm, PasswordResetForm
from .models import User


class SignupForm(UserCreationForm):
    username = forms.CharField(max_length=30, widget=forms.TextInput(attrs={'required': 'true', }))
    email = forms.EmailField(widget=forms.EmailInput(attrs={'required': 'True', }))
    mobile = forms.CharField(widget=forms.TextInput(attrs={'required': 'True', }))
    password1 = forms.CharField(widget=forms.PasswordInput(attrs={'required': 'true', }))
    password2 = forms.CharField(widget=forms.PasswordInput(attrs={'required': 'true', }))
    gender = forms.ChoiceField(widget=forms.Select(), choices=([(1, '남성'), (2, '여성')]), initial=1, required=True)
    birth = forms.DateTimeField(required=False)

    class Meta:
        model = User
        fields = ('username', 'email', 'mobile', 'password1', 'password2', 'gender', 'birth')

    def clean_password2(self):
        # Check that the two password entries match
        password1 = self.cleaned_data.get("password1")
        password2 = self.cleaned_data.get("password2")
        if password1 and password2 and password1 != password2:
            raise forms.ValidationError("패스워드를 다시 확인해주시기 바랍니다.")
        return password2


class SearchForm(UserChangeForm):
    email = forms.EmailField(widget=forms.EmailInput(attrs={'required': 'True', }))


class SigninForm(AuthenticationForm):
    email = forms.EmailField(required=True,
                             widget=forms.EmailInput(attrs={'required': 'True', }))
    password = forms.CharField(widget=forms.PasswordInput(attrs={'required': 'true', }))
    id_save = forms.CheckboxInput()
