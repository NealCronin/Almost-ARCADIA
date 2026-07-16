from django import forms


class ServiceForm(forms.Form):
    location = forms.ChoiceField(
        choices=[("local", "Local"), ("remote", "Remote")]
    )
    remote_host = forms.CharField(required=False, initial="127.0.0.1")
    instruction_port = forms.IntegerField(required=False, initial=9000)

    service_type = forms.ChoiceField(
        choices=[("llm", "LLM"), ("sam3", "SAM3")]
    )
    inference_port = forms.IntegerField(min_value=1, max_value=65535)
    settings_json = forms.JSONField(initial=dict)


class PromptForm(forms.Form):
    endpoint_host = forms.CharField(initial="127.0.0.1")
    endpoint_port = forms.IntegerField(min_value=1, max_value=65535)
    prompt = forms.CharField(widget=forms.Textarea)
