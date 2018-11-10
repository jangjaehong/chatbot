from django.contrib import admin
from .models import *


class ChatReportAdmin(admin.ModelAdmin):
    list_display = ('uid', 'contents', 'pub_date')


class ChatSequenceAdmin(admin.ModelAdmin):
    list_display = ('answer', 'question', 'pub_date')


class VocabDictAdmin(admin.ModelAdmin):
    list_display = ('idx','vocab')


admin.site.register(ChatReport, ChatReportAdmin)
admin.site.register(ChatSequence, ChatSequenceAdmin)
admin.site.register(VocabDict, VocabDictAdmin)

