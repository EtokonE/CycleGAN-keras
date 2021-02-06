#!/bin/bash
MODEL=generatorB2A.h5
URL=
ZIP_FILE=../models/$MODEL.zip
TARGET_DIR=../models/$MODEL/
BOT_TARGET_DIR=../tg_bot/static/model/$MODEL/
curl -L $URL -o $ZIP_FILE
mkdir $TARGET_DIR
unzip $ZIP_FILE -d $TARGET_DIR
mkdir $BOT_TARGET_DIR
unzip $ZIP_FILE -d $BOT_TARGET_DIR
rm $ZIP_FILE