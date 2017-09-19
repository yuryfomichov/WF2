#!/usr/bin/env bash

cd WF2
source .env/bin/activate
git pull origin master
cd src
python3 start.py
