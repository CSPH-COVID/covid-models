import os
import pandas as pd
from io import BytesIO
from os.path import split,join
from google.cloud.bigquery import Client


def process_model_forecast(client,file_list):
    match_blobs = [blob for blob in file_list if blob.name.endswith("__model_forecast.csv")]
    all_dfs = {}
    for blob in match_blobs:
        path,fname = split(blob.name)
        _,reg_name = split(path)
        tmp_buffer = BytesIO()
        client.download_blob_to_file(blob_or_uri=blob,file_obj=tmp_buffer)
        tmp_buffer.seek(0)
        df = pd.read_csv(tmp_buffer,index_col="date")
        all_dfs[reg_name] = df
    comb_df = pd.concat(all_dfs,names=["region","date"])
    return comb_df.to_csv()


def process_out2(client,file_list):
    match_blobs = [blob for blob in file_list if blob.name.endswith("_fit_out2.csv")]
    all_dfs = []
    for blob in match_blobs:
        path,fname = split(blob.name)
        _,reg_name = split(path)
        tmp_buffer = BytesIO()
        client.download_blob_to_file(blob_or_uri=blob,file_obj=tmp_buffer)
        tmp_buffer.seek(0)
        df = pd.read_csv(tmp_buffer,index_col=["region","date"])
        all_dfs.append(df)
    comb_df = pd.concat(all_dfs,names=["region","date"])
    return comb_df.to_csv()

def process_immun(client,file_list,pattern):
    match_blobs = [blob for blob in file_list if blob.name.endswith(pattern)]
    all_dfs = {}
    for blob in match_blobs:
        path,fname = split(blob.name)
        _,reg_name = split(path)
        tmp_buffer = BytesIO()
        client.download_blob_to_file(blob_or_uri=blob,file_obj=tmp_buffer)
        tmp_buffer.seek(0)
        df = pd.read_csv(tmp_buffer,index_col="date")
        all_dfs[reg_name] = df
    comb_df = pd.concat(all_dfs,names=["region","date"])
    return comb_df.to_csv()

def combine_files(project:str,bucket:str,subdir:str):
    # Make GCS client
    gcs_client = Client(project=project)
    bucket = gcs_client.get_bucket(bucket_or_name=bucket)
    comb_out_dir = "combined"
    # List all files in bucket subdirectory
    file_list = list(gcs_client.list_blobs(bucket_or_name=bucket,prefix=subdir))
    # Get files matching pattern within subdir.
    model_forecast_df_str = process_model_forecast(client=gcs_client,file_list=file_list)
    model_forecast_blob = bucket.blob(blob_name=join(subdir,comb_out_dir,"combined_model_forecast.csv"))
    model_forecast_blob.upload_from_string(model_forecast_df_str)
    # Process 1inX file
    out2_df_str = process_out2(client=gcs_client,file_list=file_list)
    out2_blob = bucket.blob(blob_name=join(subdir,comb_out_dir,"combined_model_out2.csv"))
    out2_blob.upload_from_string(out2_df_str)
    # Process immunity files
    for pattern in ["_immun_ba45.csv","_immun_65p_ba45.csv","_immun_hosp_ba45.csv","_immun_hosp_65p_ba45.csv",
                          "_immun_emv.csv","_immun_65p_emv.csv","_immun_hosp_emv.csv","_immun_hosp_65p_emv.csv"]:
        output_df_str = process_immun(client=gcs_client,file_list=file_list,pattern=pattern)
        output_fname = "combined" + pattern
        output_blob = bucket.blob(blob_name=join(subdir,comb_out_dir,output_fname))
        output_blob.upload_from_string(output_df_str)
    print("Here")

def download_files(output_dir:str,project:str,bucket:str,subdir:str):
    # Make GCS client
    gcs_client = Client(project=project)
    bucket = gcs_client.get_bucket(bucket_or_name=bucket)
    comb_out_dir = "combined"
    # List all files in bucket subdirectory
    get_patterns = {"__model_forecast.png","batch_32__model_fit.png"}
    get_file_list = [f for f in gcs_client.list_blobs(bucket_or_name=bucket, prefix=subdir) if any([f.name.endswith(patt) for patt in get_patterns])]
    # Make output directory
    os.makedirs(output_dir,exist_ok=True)
    for file in get_file_list:
        file.download_to_filename(join(output_dir,split(file.name)[-1]))


if __name__ == "__main__":
    combine_files("co-covid-models","covid-rmw-model-results","rmw-model-run-1671139866")
    download_files("batch_rmw_results","co-covid-models","covid-rmw-model-results","rmw-model-run-1671664524")

