from django.apps import AppConfig
from django.conf import settings



class AppConfig(AppConfig):
    # path - os.path.join(settings.MODELS, )
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'app'
