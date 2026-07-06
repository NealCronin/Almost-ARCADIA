"""
ASGI config for the drone orchestrator project.
"""

import os
from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "drone_orchestrator.settings")
application = get_asgi_application()
