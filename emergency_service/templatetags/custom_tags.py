   # emergency_app/emergency_service/templatetags/custom_tags.py

from django import template

register = template.Library()

@register.filter
def zip_lists(a, b):
    return zip(a, b)