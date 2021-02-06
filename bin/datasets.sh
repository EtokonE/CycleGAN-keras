#!/bin/bash
FILE=$1


if [[ $FILE == 'monet2photo' ]]; then
	URL=https://www.dropbox.com/sh/u4c5o3y2qwtipy5/AADtxxa0rcQvulDwuGOAgBHHa?dl=0
	echo $URL
elif [[ $FILE == 'portrait2photo' ]]; then
	URL=https://www.dropbox.com/sh/r738501dvqnusbk/AACKessreIZhf3i3vXs3P1_wa?dl=0
	echo $URL
else
	echo "Available datasets are: monet2photo and portrait2photo"
	exit 1
fi

ZIP_FILE=../cyclegan_datasets/$FILE.zip
TARGET_DIR=../cyclegan_datasets/$FILE/
curl -L $URL -o $ZIP_FILE
mkdir $TARGET_DIR
unzip $ZIP_FILE -d $TARGET_DIR
rm $ZIP_FILE