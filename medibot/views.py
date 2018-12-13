from django.shortcuts import render, redirect, reverse
from django.http import HttpResponse

import math
from datetime import datetime
import json

from .models import *
import algorithm.chatbot as chatbot


# 메인페이지
def index(request):
    if request.user.is_authenticated:
        # 로그인, 건강체크, 영양체크 확인
        print("check_start")
        physical_report, measure_report, intake_food_report = manage_check(request.user.pk, request.user.username, request.user.last_login)
        # 대화기록 로드
        chat_reports = ChatReport.objects.filter(uid=request.user.pk).order_by('pub_date')
        return render(request, 'medibot/index.html',
                      {"chat_reports": chat_reports,
                       "physical_report": physical_report,
                       "measure_report": measure_report,
                       "intake_food_report": intake_food_report})
    else:
        return redirect(reverse('accounts:login'))

def manage_check(uid, username, last_login):
    #페이지 로드시에 사용, 유효성 체크
    now_date = datetime.now()
    # 챗봇
    speaker = "com"
    botname = "Medi-Bot"
    contents = ""
    # 로그인 체크
    last_login_day = (now_date - last_login).days
    if last_login_day > 1:
        contents = "%s님 %d일만에 접속하셨네요." \
                   "혹시 사용법이 기억안나신다면 \'도움말\' 입력해주세요" % (username, last_login_day)
        # 보낼 메세지 저장
        ChatReport(uid=uid, speaker=speaker, username=botname, contents=contents,
                   pub_date=timezone.now()).save()

    #신체정보
    physical_report = PhysicalReport.objects.filter(uid=uid).last()
    if not physical_report:
        contents = "신체정보를 한번도 등록안하셨네요. 먼저 신체정보를 등록해주세요!"
        ChatReport(uid=uid, speaker=speaker, username=botname, contents=contents,
                   pub_date=timezone.now()).save()

    # 건강체크 확인
    measure_report = MeasureReport.objects.filter(uid=uid).last()
    if measure_report:
        last_check_day = (now_date - measure_report.pub_date).days
        # 마지막 체크일이 어제일 경우
        if last_check_day == 1:
            contents = "%s님의 어제 체질량지수: %d | %s 복부비만도: %d | %s, 기초대사량: %d | %s 였네요. 오늘도 꼭 체크 해주세요!"\
                       % (username,
                          measure_report.bmi, measure_report.bmi_state,
                          measure_report.whr, measure_report.whr_state,
                          measure_report.energy, measure_report.energy_state,)
        # 마지막 체크일이 1일 초과
        elif last_check_day > 1:
            contents = "%s님 확인해보니깐 %s 이후로 건강 체크가 하신적이 없네요." \
                       "건강체크는 매일 체크해서 관리를 해줘야 효과가 있답니다." \
                       % (username, measure_report.pub_date.strftime("%Y-%m-%d"),)
        # 보낼 메세지 저장
        ChatReport(uid=uid, speaker=speaker, username=botname, contents=contents,
                   pub_date=timezone.now()).save()
    else:
        # 체크 기록이 없음
        contents = "%s님 건강 체크를 한번도 하신적이 없네요." \
                   "그러면 안되며 만성질환은 언제 생길지 몰라요!!!" \
                   "\'건강체크 시작\'이라고 하면 제가 확인해드릴게요." \
                   % username
        # 보낼 메세지 저장
        ChatReport(uid=uid, speaker=speaker, username=botname, contents=contents,
                   pub_date=timezone.now()).save()

    # 식단체크 확인
    intake_food_report = IntakeFoodReport.objects.filter(uid=uid).last()
    if intake_food_report:
        last_check_day = (now_date - intake_food_report.pub_date).days
        if last_check_day == 1:
            contents = "%s님 어제 총섭취량: %d kcal 드셨네요.\n" \
                       "탄수화물: %d\n" \
                       "단백질: %d\n" \
                       "지방: %d\n" \
                       "당류: %d\n" \
                       "나트륨: %d\n" \
                       "콜레스테롤: %d\n" \
                       "불포화지방: %d\n" \
                       "트랜스지방: %d\n" \
                       "영양소는 이렇게 드셨네요. 오늘도 꼭 체크 해주세요!" \
                       % (username, intake_food_report.kcal,
                          intake_food_report.carbohydrate, intake_food_report.protein,
                          intake_food_report.fat, intake_food_report.sugars,
                          intake_food_report.salt, intake_food_report.cholesterol,
                          intake_food_report.saturatedfat, intake_food_report.transfat)
        # 마지막 체크일이 1일 초과
        elif last_check_day > 1:
            contents = "%s님 영양 체크를 %s 이후로 하신적이 없네요. 매일 매일 관리 해줘야 좋아요." \
                       % (username, intake_food_report.pub_date.strftime("%Y-%m-%d"),)
        # 보낼 메세지 저장
        ChatReport(uid=uid, speaker=speaker, username=botname, contents=contents,
                   pub_date=timezone.now()).save()
    else:
        # 체크 기록이 없음
        contents = "%s님 영양 체크를 한번도 하신적이 없네요.." \
                   "오늘은 드신음식을 통해 얼마나 영양소를 섭취했는지 알아보세요." \
                   "\'영양체크 시작\'이라고 하면 제가 확인해드릴게요." \
                   % username
        # 보낼 메세지 저장
        ChatReport(uid=uid, speaker=speaker, username=botname, contents=contents,
                   pub_date=timezone.now()).save()
    return physical_report, measure_report, intake_food_report


def save_chatting(request):
    if request.user.is_authenticated:
        if request.method == 'POST':
            if request.is_ajax():
                uid = request.user.pk
                speaker = request.POST['speaker']
                username = request.POST['username']
                contents = request.POST['contents']
                ChatReport(uid=uid, speaker=speaker, username=username, contents=contents, pub_date=timezone.now()).save()
    return render(request)


def physical_update(request):
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
                # 정보 업데이트
                PhysicalReport(uid=uid, age=age, gender=gender, stature=stature, weight=weight, waist=waist, hip=hip, pub_date=timezone.now()).save()
                physical_info = PhysicalReport.objects.filter(uid=request.user.pk).last()
                return render(request, 'medibot/user_info.html', {"physical_report": physical_info})
        return render(request)
    else:
        return redirect(reverse('accounts:login'))


def comunication(request):
    if request.user.is_authenticated:
        if request.method == 'POST':
            if request.is_ajax():
                #사용자 입력 대화 저장
                msg = request.POST['msg']
                uid = request.user.pk
                ChatReport(uid=uid, speaker='user', username=request.user.username, contents=msg, pub_date=timezone.now()).save()
                # 사용자 질의문에 대한 챗봇의 답변
                reply = chatbot._get_answer(msg)
                funcIdx = reply.rfind('fn')
                if funcIdx:
                    answer = reply[:funcIdx]
                    func = reply[funcIdx+4:].strip()
                else:
                    answer = reply
                    func = ""
                ChatReport(uid=uid, speaker='com', username='Medi-BOT', contents=answer, pub_date=timezone.now()).save()
                # 답변 반환
                context = [{'message': answer, 'func': func, 'name': 'Medi-BOT'}]
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
    return render(request)


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
        energy_state = '?'
        if gender == 1:
            energy_result = (66.47 + (13.75 * weight) + (5 * stature) - (6.76 * age))
            if age >=20 or age <= 29:
                if energy_result >= 1359.8 or energy_result < 1728:
                    energy_state = '낮음'
                elif energy_result >= 1728 or energy_result <= 2096.2:
                    energy_state = '적정'
                elif energy_result > 2096.2:
                    energy_result = '높음'
            if age >= 30 or age <= 49:
                if energy_result >= 1367.4 or energy_result < 1669.5:
                    energy_state = '낮음'
                elif energy_result >= 1669.5 or energy_result <= 1971.6:
                    energy_state = '적정'
                elif energy_result > 1971.6:
                    energy_state = '높음'
            if age >= 50:
                if energy_result >= 1178.5 or energy_result < 1493.8:
                    energy_state = '낮음'
                elif energy_result >= 1493.8 or energy_result <= 1809.1:
                    energy_state = '적정'
                elif energy_result > 1809.1:
                    energy_state = '높음'
        else:
            energy_result = (65.51 + (9.56 * weight) + (1.85 * stature) - (4.68 * age))
            if age >=20 or age <= 29:
                if energy_result >= 1078.5 or energy_result < 1311.5:
                    energy_state = '낮음'
                elif energy_result >= 1311.5 or energy_result <= 1544.5:
                    energy_state = '적정'
                elif energy_result > 1544.5:
                    energy_result = '높음'
            if age >= 30 or age <= 49:
                if energy_result >= 1090.9 or energy_result < 1316.8:
                    energy_state = '낮음'
                elif energy_result >= 1316.8 or energy_result <= 1542.7:
                    energy_state = '적정'
                elif energy_result > 1542.7:
                    energy_state = '높음'
            if age >= 50:
                if energy_result >= 1023.9 or energy_result < 1252.5:
                    energy_state = '낮음'
                elif energy_result >= 1252.5 or energy_result <= 1481.1:
                    energy_state = '적정'
                elif energy_result > 1481.1:
                    energy_state = '높음'
        return energy_result, energy_state


def day_measure(request):
    # 측정 성공:1, 신체정보 조회 실패:2
    if request.user.is_authenticated:
        if request.method == 'POST':
            if request.is_ajax():
                physical_report = PhysicalReport.objects.filter(uid=request.user.pk).last()
                if physical_report:
                    uid = request.user.pk
                    age = physical_report.age
                    gender = physical_report.gender
                    stature = physical_report.stature
                    weight = physical_report.weight
                    waist = physical_report.waist
                    hip = physical_report.hip

                    # bmi 계산
                    calc = Calc()
                    bmi_result, bmi_state = calc.bmi(stature, weight)
                    whr_result, whr_state = calc.whr(gender, waist, hip)
                    energy_result, energy_state = calc.energy(gender, age, stature, weight)
                    # 기록 DB 저장
                    MeasureReport(uid=uid, bmi=bmi_result, bmi_state=bmi_state, whr=whr_result, whr_state=whr_state,
                                  energy=energy_result, energy_state=energy_state, pub_date=timezone.now()).save()
                    BmiReport(uid=uid, stature=stature, weight=weight, bmi=bmi_result, state=bmi_state, pub_date=timezone.now()).save()
                    WHRReport(uid=uid, gender=gender, waist=waist, hip=hip, whr=whr_result, state=whr_state, pub_date=timezone.now()).save()
                    EnergyReport(uid=uid, gender=gender, age=age, stature=stature, weight=weight, energy=energy_result, state=energy_state, pub_date=timezone.now()).save()
                    # 리턴값
                    context = {'bmi': bmi_result, 'bmi_state': bmi_state,
                               'whr': whr_result, 'whr_state': whr_state,
                               'energy': energy_result, 'energy_state': energy_state,
                               'age': age, 'gender': gender, "result": 1}
                    return HttpResponse(json.dumps(context), content_type="application/json")
                else:
                    return render(request, 'medibot/index.html', {"result": 2})
        return render(request)
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
        return render(request)
    else:
        return redirect(reverse('accounts:login'))


def hist(request):
    if request.user.is_authenticated:
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
        return render(request)
    else:
        return redirect(reverse('accounts:login'))



