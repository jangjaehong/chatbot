from django.shortcuts import render, HttpResponse, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.sites.shortcuts import get_current_site
from django.template.loader import render_to_string
from django.core.mail import EmailMessage
from django.utils.encoding import force_bytes, force_text
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode

from .forms import SignupForm, SigninForm, SearchForm
from .token import account_activation_token
from .models import User


def activate(request, uidb64, token):
    try:
        uid = force_text(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)
    except(TypeError, ValueError, OverflowError):
        user = None
    if user is not None and account_activation_token.check_token(user, token):
        user.is_activate = True
        user.save()
        login(request, user)
        messages.success(request, "회원가입을 축하드립니다.")
        return redirect('medibot:index')
    else:
        return HttpResponse("이메일 링크를 확인해주세요.")


def signup(request):
    signupform = SignupForm()
    if request.method == "POST":
        signupform = SignupForm(request.POST)
        if signupform.is_valid():
            new_user = User.objects.create_user(signupform.cleaned_data)
            #user = signupform.save(commit=False)
            #user.is_active = False
            new_user.save()
            #
            # current_site = get_current_site(request)
            # message = render_to_string('user/account_activate_email.html', {
            #     'user': user,
            #     'domain': current_site,
            #     'uid': urlsafe_base64_encode(force_bytes(user.pk)).decode('utf-8'),
            #     'token': account_activation_token(user)
            # })
            #
            # """이메일전송"""
            # mail_subject = 'test'
            # to_email = signupform.cleaned_data.get('email')
            # email = EmailMessage(mail_subject, message, to=[to_email])
            # email.send()
            #
            messages.success(request, "가입시 입력한 이메일로 인증메일이 발송되었습니다.\n"
                                      "이메일 인증 후 서비스를 이용하실 수 있습니다.")
            return redirect('accounts:login')
        else:
            messages.error(request, "이메일이나 휴대폰번호가 이미 존재합나디.")
            return redirect('accounts:login')
    else:
        return render(request, "user/join.html", {"signupform": signupform,})


def search(request):
    searchForm = SearchForm()
    if request.method == "POST":
        email = request.POST["email"]
    else:
        return render(request, "user/search.html", {"searchForm": searchForm, })


def signin(request):
    # 로그인 기본 폼 로드
    request.session.flush()
    signinform = SigninForm()
    if request.method == "POST":
        email = request.POST["email"]
        password = request.POST["password"]
        user = authenticate(email=email, password=password)
        if user is not None:
            login(request, user)
            return redirect('medibot:index')
        else:
            messages.error(request, "정보가 일치하지 않습니다.\n이메일주소나 패스워드를 다시 확인해보시기 바랍니다.")
            return redirect('accounts:login')
    else:
        messages.info(request, "로그인을 하셔야 서비스를 이용 하실수 있습니다.")
        return render(request, "user/login.html", {"signinform": signinform, })


def signout(request):
    logout(request)
    return redirect('accounts:login')
