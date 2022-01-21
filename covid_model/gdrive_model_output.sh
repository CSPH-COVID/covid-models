#/bin/bash

# purpose is to automatically execute run_additional_outputs.py and upload files to gdrive every saturday

# uses GDrive tool from https://github.com/prasmussen/gdrive

# model repo is located at 
# /var/covid/covid-models/

# python script is located at
# # /var/covid/covid-models/covid_model

# model output is located at 
# /var/covid/covid-models/covid_model/output/

# primary gdrive folder is located at
# https://drive.google.com/drive/u/0/folders/1ygc1-aB1JP3rLEpbyJ6sRJ9Q1v1gsfor

# within gdrive folder, 3 output files should be uploaded 
# and, within "Daily Archive" sub folder, create a dated subfolder with the same files
# "Daily Archive" sub folder located at https://drive.google.com/drive/u/0/folders/1FGTjKIVwHba-oMoLnFALdz8_ZaRLZjCv



# step 1 - execute python and generate output files
# step 2  - upload files to main gdrive folder
# step 3 - create dated sub folder and upload there, too

LOCAL_PATH=/var/covid/covid-models/covid_model/

# change to temp directory
cd $LOCAL_PATH

# FOLDER_TITLE date 
FOLDER_TITLE=$(date +%Y-%m-%d)


# clean out any existing output files first 
# assumes csv suffix 
rm $LOCAL_PATH/output/*.csv

# execute python to generate output files 
/usr/bin/python3.7  $LOCAL_PATH/run_additional_outputs.py


# loop through putput files
for FILENAME in $LOCAL_PATH/output/*.csv; do

	# gdrive upload/sync
	# main folder ID 1ygc1-aB1JP3rLEpbyJ6sRJ9Q1v1gsfor
	# uploads everything in the output dir 
	/etc/gdrive/gdrive --service-account $GD_SERVICE_ACCOUNT upload --parent 1ygc1-aB1JP3rLEpbyJ6sRJ9Q1v1gsfor $FILENAME 


done 


# now create a dated sub folder, move files into it and sync that to additional gdrive folder 
mkdir $LOCAL_PATH/output/$FOLDER_TITLE
# move into sub folder
mv $LOCAL_PATH/output/*.csv $LOCAL_PATH/output/$FOLDER_TITLE
# upload
/etc/gdrive/gdrive --service-account $GD_SERVICE_ACCOUNT upload -r --parent 1FGTjKIVwHba-oMoLnFALdz8_ZaRLZjCv $LOCAL_PATH/output/$FOLDER_TITLE

# now remove the temp dated subfolder from output 
rm -Rf $LOCAL_PATH/output/$FOLDER_TITLE