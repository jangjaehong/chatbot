from django.contrib import admin
from .models import *


class ChatReportAdmin(admin.ModelAdmin):
    list_display = ('uid', 'contents', 'pub_date')


class ChatSequenceAdmin(admin.ModelAdmin):
    list_display = ('answer', 'question', 'pub_date')


class VocabDictAdmin(admin.ModelAdmin):
    list_display = ('idx','vocab')


class PhysicalAdmin(admin.ModelAdmin):
    list_display = ('uid', 'age', 'gender', 'stature', 'weight', 'waist', 'pub_date')


class MeasureAdmin(admin.ModelAdmin):
    list_display = ('uid', 'bmi', 'bmi_state', 'whr', 'whr_state', 'energy', 'energy_state', 'pub_date')


class IntakeFoodAdmin(admin.ModelAdmin):
    list_display = ('uid', 'foodlist', 'energy', 'gram', 'kcal', 'carbohydrate', 'protein', 'fat',
                    'sugars', 'salt', 'cholesterol', 'saturatedfat', 'transfat', '', 'pub_date')


admin.site.register(ChatReport, ChatReportAdmin)
admin.site.register(ChatSequence, ChatSequenceAdmin)
admin.site.register(VocabDict, VocabDictAdmin)
admin.site.register(PhysicalReport, PhysicalAdmin)
admin.site.register(MeasureReport, MeasureAdmin)
admin.site.register(IntakeFoodReport, IntakeFoodAdmin)
