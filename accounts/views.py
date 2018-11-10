from django.shortcuts import render, HttpResponse, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from .forms import SignupForm, SigninForm

def signup(request):
    signupform = SignupForm()
    if request.method == "POST":
        signupform = SignupForm(request.POST)
        if signupform.is_valid():
            user = signupform.save(commit=False)
            user.email = signupform.cleaned_data['email']
            user.mobile = signupform.cleaned_data['mobile']
            user.password = signupform.clean_password2()
            user.save()

            messages.success(request, "회원가입을 축하드립니다.\n 로그인을 하시면 서비스를 이용하실수 있습니다.")
            return redirect('accounts:login')
        else:
            messages.error(request, "이메일이나 휴대폰번호가 이미 존재합나디.")
            return redirect('accounts:login')
    else:
        return render(request, "user/join.html", {"signupform": signupform,})


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