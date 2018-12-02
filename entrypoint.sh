#!/bin/sh
export DJANGO_SETTINGS_MODULE="AnyTimeGridSearchCV.settings.docker_settings"
python -W ignore manage.py makemigrations
python -W ignore manage.py migrate
python -W ignore manage.py iris_experiments Tree IRIS
exec "$@"
