#!/bin/bash

coverage run -m pytest src/
coverage report -m
