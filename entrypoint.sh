#!/bin/sh
export DJANGO_SETTINGS_MODULE="AnyTimeGridSearchCV.settings.docker_settings"
python manage.py makemigrations
python manage.py migrate
python manage.py iris_experiments Tree IRIS
exec "$@"
