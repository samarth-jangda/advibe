#!/bin/bash
# listing all necessary paths
# NOTE: The input_path will be pre-prepared

# required paths
models_data="/home/samarth-jangda/development/models_data"
python_env_name="advibe_env"
data_path="/home/samarth-jangda/development/advibe/modules/data" #------> parent data path
module_data="$data_path/module" #---------------------------------------> path for all modules general and domain data
general="$module_data/general" #----------------------------------------> path for general data directory in module folder
domain="$module_data/domain" #------------------------------------------> path for domain data directory in module folder
input_path="$data_path/inference/input" #-------------------------------> input path for uploading data for inference 
output_path="$data_path/inference/output" #-----------------------------> output path of prediction of data in input_path
output_model_path="$data_path/lib" #------------------------------------> path for saving trained models
detr_path="$general/detr" #---------------------------------------------> detection transformer data path
stable_diffusion_path="$general/stable_diffusion" #---------------------> stable diffusion data path
prompts_data_path="$general/prompts" #----------------------------------> path to store all prompts data
