# -*- coding: utf-8 -*-
# Generated by Django 1.11.1 on 2017-05-09 19:07
from __future__ import unicode_literals

import django.contrib.postgres.fields.jsonb
from django.db import migrations, models
import django.db.models.deletion
import uuid


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='CVResult',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('score', models.FloatField()),
                ('params', django.contrib.postgres.fields.jsonb.JSONField(default=dict)),
            ],
        ),
        migrations.CreateModel(
            name='GridSearch',
            fields=[
                ('uuid', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('classifier', models.CharField(max_length=128)),
            ],
        ),
        migrations.AddField(
            model_name='cvresult',
            name='gridsearch',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='grids.GridSearch'),
        ),
    ]
