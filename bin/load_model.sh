#!/bin/bash
MODEL=generatorB2A.h5
URL=https://www.dropbox.com/s/sg4pj0dvaydp6ey/generatorB2A.h5?dl=0
TARGET_DIR=./models/$MODEL
BOT_TARGET_DIR=./tg_bot/static/model/$MODEL
curl -L $URL -o $ZIP_FILE
cp $TARGET_DIR $BOT_TARGET_DIR
