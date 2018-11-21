from django.shortcuts import render, redirect, reverse
from django.http import HttpResponse
from django.utils import timezone

import math
import datetime
import json

from .models import *
import algorithm.chatbot as chatbot

def test(request):
    return render(request, 'medibot/test.html')


# 화면이나 데이터 렌더링 하는 공간
def index(request):
    if request.user.is_authenticated:
        chatreport = ChatReport.objects.filter(uid=request.user.pk).order_by('pub_date')
        physical_report = PhysicalReport.objects.filter(uid=request.user.pk).order_by('pub_date')[:1]

        return render(request, 'medibot/index.html', {"chatreport": chatreport, "physical_report": physical_report})
    else:
        return redirect(reverse('accounts:login'))


def ajaxpost(request):
    if request.user.is_authenticated:
        if request.method == 'POST':
            if request.is_ajax():
                #사용자 입력 대화 저장
                msg = request.POST['msg']
                uid = request.user.pk
                ChatReport(uid=uid, speaker='user', username=request.user.username, contents=msg, pub_date=timezone.now()).save()

                #사용자 입력 대화를 통한 컴퓨터의 답변 찾기

                reply = chatbot._get_answer(msg)
                print(reply)
                ChatReport(uid=uid, speaker='com', username='Medi-BOT', contents=reply, pub_date=timezone.now()).save()

                # 호출한 곳으로 리턴, json은 꼭 이렇게
                context = [{'message': reply, 'name': 'Medi-BOT'}]
                return HttpResponse(json.dumps(context), content_type="application/json")
        return render(request)
    else:
        return redirect(reverse('accounts:login'))


def autocomplete(request):
    if request.method == 'POST':
        if request.is_ajax():
            value = request.POST['value']
            foodlist = FoodList.objects.filter(name__startswith=value)
            context = []
            for food in foodlist:
                pk = food.pk
                name = food.name
                gram = food.onegram
                kcal = food.kcal
                context.append({'pk': pk, 'name': name, 'kcal': kcal, 'gram': gram})
            return HttpResponse(json.dumps(context), content_type="application/json")
    return render(request, 'medibot/index.html')


# 신체측정
class Calc:
    def bmi(self, stature, weight):
        stature_pow = math.pow(stature / 100, 2)
        bmi_result = weight / stature_pow
        bmi_result = round(bmi_result, 3)

        if bmi_result >= 35.0:
            bmi_state = '고도비만'
        elif (bmi_result >= 30.0) and (bmi_result <= 35.0):
            bmi_state = '중등도 비만'
        elif (bmi_result >= 25.0) and (bmi_result <= 30.0):
            bmi_state = '경도 비만'
        elif (bmi_result >= 23.0) and (bmi_result <= 24.9):
            bmi_state = '과체중'
        elif (bmi_result >= 18.5) and (bmi_result <= 22.9):
            bmi_state = '정상'
        elif bmi_result > 18.5:
            bmi_state = '저체중'
        else:
            bmi_state = '측정불가'
        return bmi_result, bmi_state

    def whr(self, gender, waist, hip):
        whr_result = waist/hip
        whr_result = round(whr_result, 4)

        whr_state = '정상'
        if gender == 1:
            if whr_result >= 0.95:
                whr_state = '복부비만'
        else:
            if whr_result >= 0.85:
                whr_state = '복부비만'
        return whr_result, whr_state

    def energy(self, gender, age, stature, weight):
        energy_result = 0.0
        if gender == 1:
            energy_result = (66.47 + (13.75 * weight) + (5 * stature) - (6.76 * age))
        else:
            energy_result = (65.51 + (9.56 * weight) + (1.85 * stature) - (4.68 * age))
        return energy_result


def physical(request):
    if request.user.is_authenticated:
        if request.method == 'POST':
            if request.is_ajax():
                uid = request.user.pk
                age = int(request.POST['age'])
                gender = int(request.POST['gender'])
                stature = float(request.POST['stature'])
                weight = float(request.POST['weight'])
                waist = float(request.POST['waist'])
                hip = float(request.POST['hip'])

                # bmi 계산
                calc = Calc()
                bmi_result, bmi_state = calc.bmi(stature, weight)
                whr_result, whr_state = calc.whr(gender, waist, hip)
                energy = calc.energy(gender, age, stature, weight)
                # 기록 DB 저장
                PhysicalReport(uid=uid, age=age, gender=gender,
                               stature=stature, weight=weight,
                               waist=waist, hip=hip,
                               bmi=bmi_result, bmi_state=bmi_state,
                               whr=whr_result, whr_state=whr_state,
                               energy=energy, energy_state="", pub_date=timezone.now()).save()
                # 리턴값
                context = {'bmi': bmi_result, 'bmi_state': bmi_state,
                           'whr': whr_result, 'whr_state': whr_state,
                           'energy': energy, 'energy_state': "평균",
                           'age': age, 'gender': gender}
                return HttpResponse(json.dumps(context), content_type="application/json")
        else:
            chatreport = ChatReport.objects.filter(uid=request.user.pk).order_by('pub_date')
            context = {'chatreport': chatreport}
            return render(request, 'medibot/index.html', context)
    else:
        return redirect(reverse('accounts:login'))


def diet(request):
    if request.user.is_authenticated:
        if request.method == 'POST':
            if request.is_ajax():
                uid = request.user.pk
                count = int(request.POST['fcount'])
                fname = request.POST.getlist("fname")
                pks = request.POST.getlist('fpk')
                servings = request.POST.getlist('fserving')

                context = {"energy": 0.0, "gram": 0.0, "kcal": 0.0, "carbohydrate": 0.0, "protein": 0.0, "fat": 0.0,
                           "sugars": 0.0, "salt": 0.0, "cholesterol": 0.0, "saturatedfat": 0.0, "transfat": 0.0}

                # 이용자 마지막으로 기록된 기초대사량 가져옴
                row = PhysicalReport.objects.filter(uid=uid).order_by('pub_date')
                if row is not None:
                    if float(row[0].energy) is not 0:
                        context["energy"] = float(row[0].energy)
                        for i in range(count):
                            rows = FoodList.objects.filter(pk=pks[i])
                            for row in rows:
                                context["gram"] += (float(row.onegram) * float(servings[i]))
                                context["kcal"] += (float(row.kcal) * float(servings[i]))
                                context["carbohydrate"] += (float(row.carbohydrate) * float(servings[i]))
                                context["protein"] += (float(row.protein) * float(servings[i]))
                                context["fat"] += (float(row.fat) * float(servings[i]))
                                context["sugars"] += (float(row.sugars) * float(servings[i]))
                                context["salt"] += (float(row.salt) * float(servings[i]))
                                context["cholesterol"] += (float(row.cholesterol) * float(servings[i]))
                                context["saturatedfat"] += (float(row.saturatedfat) * float(servings[i]))
                                context["transfat"] += (float(row.transfat) * float(servings[i]))

                        # DB저장
                        IntakeFoodReport(uid=uid, foodlist=fname, energy=context["energy"], gram=context["gram"], kcal=context["kcal"],
                                         carbohydrate=context["carbohydrate"], protein=context["protein"], fat=context["fat"],
                                         sugars=context["sugars"], salt=context["salt"], cholesterol=context["cholesterol"],
                                         saturatedfat=context["saturatedfat"], transfat=context["transfat"], pub_date=timezone.now()).save()
                return HttpResponse(json.dumps(context), content_type="application/json")
        else:
            chatreport = ChatReport.objects.filter(uid=request.user.pk).order_by('pub_date')
            context = {'chatreport': chatreport}
            return render(request, 'medibot/index.html', context)
    else:
        return redirect(reverse('accounts:login'))


def hist(request):
    if request.method == 'POST':
        if request.is_ajax():
            # 데이터베이스에서 조회한 bmi 수치와 날짜 저장
            context = []
            # 조회 기준 날짜
            type = int(request.POST['type'])
            term = int(request.POST['term'])
            gubun = int(request.POST['gubun'])
            now_date = timezone.now()
            if type == 1:
                for i in range(0, term):
                    join_date = now_date + timezone.timedelta(days=-i)
                    format_date = datetime.datetime.strftime(join_date, "%Y-%m-%d")
                    row = PhysicalReport.objects.order_by("pub_date").filter(pub_date__gt=join_date)[:1]
                    if row:
                        for report in row:
                            if gubun == 1:
                                context.append({"value": report.bmi, "text": report.bmi_state, "date": format_date, "age": report.age, "gender": report.gender})
                            if gubun == 2:
                                context.append({"value": report.whr, "text": report.whr_state, "date": format_date, "age": report.age, "gender": report.gender})
                            if gubun == 3:
                                context.append({"value": report.energy, "text": report.energy_state, "date": format_date, "age": report.age, "gender": report.gender})
                    else:
                        context.append({"value": 0,  "text": "기록X", "date": format_date})
            # if type == 2:
            #     for i in range(0, term):
            #         join_date = now_date + timezone.timedelta(days=-i)
            #         format_date = datetime.datetime.strftime(join_date, "%Y-%m-%d")
            #         row = IntakeFoodReport.objects.filter(pub_date__gt=join_date).last()
            #         if row is not None:
            #             context.append({"value": row.kcal, "date": format_date})
            return HttpResponse(json.dumps(context), content_type="application/json")
    else:
        chatreport = ChatReport.objects.filter(uid=request.user.pk).order_by('pub_date')
        context = {'chatreport': chatreport}
        return render(request, 'medibot/index.html', context)



