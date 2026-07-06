"""
WSGI config for the drone orchestrator project.
"""

import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "drone_orchestrator.settings")
application = get_wsgi_application()
