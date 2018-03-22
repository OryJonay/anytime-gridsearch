'''
Created on March 22, 2018

Settings for running project with docker

@author: Ory Jonay
'''

from AnyTimeGridSearchCV.settings.base import *

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'anytimegridsearch',
        'USER': 'postgres',
        'HOST': 'db',                      # Empty for localhost through domain sockets or '127.0.0.1' for localhost through TCP.
        'PORT': 5432,
    }
}