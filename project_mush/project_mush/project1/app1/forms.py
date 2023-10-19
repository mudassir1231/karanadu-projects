from django import forms
from .models import InputData

class InputDataForm(forms.ModelForm):
    class Meta:
        model = InputData
        fields = '__all__'  # Use all fields from the model
