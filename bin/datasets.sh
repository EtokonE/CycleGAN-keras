#!/bin/bash
FILE=$1


if [[ $FILE == 'monet2photo' ]]; then
	URL=https://www.dropbox.com/sh/nm7uttg933a586y/AABgesHRCJ9ueUCAHrHVrjFTa?dl=0
	echo $URL
elif [[ $FILE == 'portrait2photo' ]]; then
	URL=https://www.dropbox.com/sh/a900c04psizk86i/AAB3I1L6tT3_lscoxIeL2C5Va?dl=0
	echo $URL
else
	echo "Available datasets are: monet2photo and portrait2photo"
	exit 1
fi

ZIP_FILE=./cyclegan_datasets/$FILE.zip
TARGET_DIR=./cyclegan_datasets/$FILE/
curl -L $URL -o $ZIP_FILE
mkdir $TARGET_DIR
unzip $ZIP_FILE -d $TARGET_DIR
rm $ZIP_FILE