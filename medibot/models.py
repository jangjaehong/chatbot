from django.db import models
from django.utils import timezone
from django.conf import settings

class Article(models.Model):
    author = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,)


# 채팅 기록
class ChatReport(models.Model):
    uid = models.IntegerField(null=False, default=0)
    username = models.CharField(max_length=10, null=False, default='')
    speaker = models.CharField(max_length=10, null=False, default='')
    contents = models.TextField(null=False, default='')
    pub_date = models.DateTimeField('등록일', default=timezone.now)
    objects = models.Manager()

    class Meta:
        verbose_name_plural = "대화기록"


class ChatSequence(models.Model):
    question = models.TextField('질문', unique=True, default='')
    answer = models.TextField('답변', unique=True, default='')
    pub_date = models.DateTimeField('등록일', default=timezone.now)

    class Meta:
        verbose_name_plural = "질의응답"


class VocabDict(models.Model):
    idx = models.IntegerField(null=False, default=0, unique=True)
    vocab = models.CharField('단어', max_length=50, unique=True)
    morpheme = models.CharField('형태소', max_length=50)

    class Meta:
        verbose_name_plural = "어휘 사전"


class PhysicalReport(models.Model):
    uid = models.IntegerField(null=False, default=0)
    age = models.IntegerField(null=False, default=0)
    gender = models.IntegerField(null=False, default=0)
    stature = models.FloatField(null=False, default=0.0)
    weight = models.FloatField(null=False, default=0.0)
    waist = models.FloatField(null=False, default=0.0)
    hip = models.FloatField(null=False, default=0.0)
    bmi = models.FloatField(null=False, default=0.0)
    bmi_state = models.CharField(max_length=10, null=False, default='')
    whr = models.FloatField(null=False, default=0.0)
    whr_state = models.CharField(max_length=10, null=False, default='')
    energy = models.FloatField(null=False, default=0.0)
    energy_state = models.CharField(max_length=10, null=False, default='')
    pub_date = models.DateTimeField('등록일', default=timezone.now)
    objects = models.Manager()

    class Meta:
        verbose_name_plural = "일일체크"


class IntakeFoodReport(models.Model):
    uid = models.IntegerField(null=False, default=0)
    foodlist = models.CharField(max_length=500, null=False, default='')
    energy = models.FloatField(null=False, default=0.0)
    gram = models.IntegerField(null=False, default=0)
    kcal = models.FloatField(null=False, default=0.0)
    carbohydrate = models.FloatField(null=False, default=0.0)
    protein = models.FloatField(null=False, default=0.0)
    fat = models.FloatField(null=False, default=0.0)
    sugars = models.FloatField(null=False, default=0.0)
    salt = models.FloatField(null=False, default=0.0)
    cholesterol = models.FloatField(null=False, default=0.0)
    saturatedfat = models.FloatField(null=False, default=0.0)
    transfat = models.FloatField(null=False, default=0.0)
    pub_date = models.DateTimeField('등록일', default=timezone.now)
    objects = models.Manager()

    class Meta:
        verbose_name_plural = "섭취음식"


class BmiReport(models.Model):
    uid = models.IntegerField(null=False, default=0)
    stature = models.FloatField(null=False, default=0.0)
    weight = models.FloatField(null=False, default=0.0)
    bmi = models.FloatField(null=False, default=0.0)
    state = models.CharField(max_length=10, null=False, default='')
    pub_date = models.DateTimeField('등록일', default=timezone.now)
    objects = models.Manager()

    class Meta:
        verbose_name_plural = "체질량지수"


class WHRReport(models.Model):
    uid = models.IntegerField(null=False, default=0)
    gender = models.IntegerField(null=False, default=0)
    waist = models.FloatField(null=False, default=0.0)
    hip = models.FloatField(null=False, default=0.0)
    whr = models.FloatField(null=False, default=0.0)
    state = models.CharField(max_length=10, null=False, default='')
    pub_date = models.DateTimeField('등록일', default=timezone.now)

    class Meta:
        verbose_name_plural = "복부비만도"


class EnergyReport(models.Model):
    uid = models.IntegerField(null=False, default=0)
    gender = models.IntegerField(null=False, default=0)
    age = models.IntegerField(null=False, default=0)
    stature = models.FloatField(null=False, default=0.0)
    weight = models.FloatField(null=False, default=0.0)
    energy = models.FloatField(null=False, default=0.0)
    state = models.CharField(max_length=10, null=False, default='')
    pub_date = models.DateTimeField('등록일', default=timezone.now)

    class Meta:
        verbose_name_plural = "기초대사량"


class FoodList(models.Model):
    type = models.CharField(max_length=500, null=False, default='')
    name = models.CharField(max_length=500, null=False, default='')
    onegram = models.IntegerField(null=False, default=0)
    kcal = models.FloatField(null=False, default=0.0)
    carbohydrate = models.FloatField(null=False, default=0.0)
    protein = models.FloatField(null=False, default=0.0)
    fat = models.FloatField(null=False, default=0.0)
    sugars = models.FloatField(null=False, default=0.0)
    salt = models.FloatField(null=False, default=0.0)
    cholesterol = models.FloatField(null=False, default=0.0)
    saturatedfat = models.FloatField(null=False, default=0.0)
    transfat = models.FloatField(null=False, default=0.0)
    objects = models.Manager()

    class Meta:
        verbose_name_plural = "식품영양"
