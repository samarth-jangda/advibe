#!/bin/bash

# The following shell script is used to automate the pipeline
# of uploading the image to the e2e server in a specified amount of time

. ./path.sh
# setting all parameters
duration=36000 #---------------------------------------------------> For 10 hours
image_name="samarth1999/aer-model" #-------------------------------> Docker Image name
container_name="aer-model-container" #-----------------------------> Docker Container name
docker_username="samarth1999" #------------------------------------> DockerHub username
docker_password="DevAcrobat@1999" #--------------------------------> DockerHub password
dockerfile_path="/opt/SpeechEmotionRecognition/Dockerfile" # ------> docker file path
image_export_path="$automation_path/model.tar" # ------------------> path to save the build docker image
ssh_key_name=id_rsa #----------------------------------------------> name of ssh rsa filename to be uploaded on created node

# preparing the image
echo "Logging to dockerhub"
echo $docker_password | docker login -u $docker_username --password-stdin
echo "Building the docker image"
sudo docker build -t $image_name -f $dockerfile_path .
echo "Pushing the docker image to dockerhub"
sudo docker push $image_name
echo "Saving the docker image to $image_export_path"
sudo docker save -o "$image_export_path" "$image_name"
echo "The size of the saved docker image is:"
ls -lh "$image_export_path"
echo "Now deploying the docker container on dockerhub"

[ ! -d "$automation_path/ssh_keys" ] && echo "Preparing the directory for saving ssh keys" && mkdir "$automation_path/ssh_keys"

# preparing the config file for starting the server via python script
cat <<EOF > "$automation_path/automation.json"
{
    "description": "The following parameters are to automate the process of making new node and running the image",
    "image_name": "$image_name",
    "ssh_key_path": "/opt/ssh_keys/id_rsa",
    "operating_system": "Ubuntu",
    "gpu_required": False,
    "delete_node": True,
    "min_node_price": 20,
    "max_node_price": 100,
    "cpu_node_price": 3,
    "gpu_specs_list": {"gpu_memory":"16","ram":50,"cpu":12,"disk_space":900},
    "cpu_specs_list": {"ram":8,"cpu":4,"disk_space":"100"}
}
EOF

cat <<EOF > "$automation_path/auth.json"
{
    "description": "The following keys are the authorization parameters",
    "api_key": "49e552ff-2a36-4fbc-866d-cb00626f6cd0",
    "project_id": 33427,
    "registry_url":"registry.e2enetworks.net",
    "auth_token": "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJGSjg2R2NGM2pUYk5MT2NvNE52WmtVQ0lVbWZZQ3FvcXRPUWVNZmJoTmxFIn0.eyJleHAiOjE3NDc5Nzc4MTMsImlhdCI6MTcxNjQ0MTgxMywianRpIjoiMTliNTFmNDgtODJmNy00MTA4LWFhODctMWQ0OGYyZjAzNDZhIiwiaXNzIjoiaHR0cDovL2dhdGV3YXkuZTJlbmV0d29ya3MuY29tL2F1dGgvcmVhbG1zL2FwaW1hbiIsImF1ZCI6ImFjY291bnQiLCJzdWIiOiIyZDA2MTQwZi0zNDVhLTQ4YzktYTA3MC04NGFmZGYyNDRhNzYiLCJ0eXAiOiJCZWFyZXIiLCJhenAiOiJhcGltYW51aSIsInNlc3Npb25fc3RhdGUiOiJiMGQxYjFkMi05MjcwLTQ1ZGUtODI3Zi01YzhhNDE1ZmU4YWEiLCJhY3IiOiIxIiwiYWxsb3dlZC1vcmlnaW5zIjpbIiJdLCJyZWFsbV9hY2Nlc3MiOnsicm9sZXMiOlsib2ZmbGluZV9hY2Nlc3MiLCJ1bWFfYXV0aG9yaXphdGlvbiIsImFwaXVzZXIiLCJkZWZhdWx0LXJvbGVzLWFwaW1hbiJdfSwicmVzb3VyY2VfYWNjZXNzIjp7ImFjY291bnQiOnsicm9sZXMiOlsibWFuYWdlLWFjY291bnQiLCJtYW5hZ2UtYWNjb3VudC1saW5rcyIsInZpZXctcHJvZmlsZSJdfX0sInNjb3BlIjoicHJvZmlsZSBlbWFpbCIsInNpZCI6ImIwZDFiMWQyLTkyNzAtNDVkZS04MjdmLTVjOGE0MTVmZThhYSIsImVtYWlsX3ZlcmlmaWVkIjpmYWxzZSwibmFtZSI6IlNhbWFydGggSmFuZ2RhIiwicHJpbWFyeV9lbWFpbCI6InNpbmdodmluYXlkZXYzMjFAZ21haWwuY29tIiwiaXNfcHJpbWFyeV9jb250YWN0IjpmYWxzZSwicHJlZmVycmVkX3VzZXJuYW1lIjoic2FtYXJ0aC5qYW5nZGFAZ21haWwuY29tIiwiZ2l2ZW5fbmFtZSI6IlNhbWFydGgiLCJmYW1pbHlfbmFtZSI6IkphbmdkYSIsImVtYWlsIjoic2FtYXJ0aC5qYW5nZGFAZ21haWwuY29tIn0.Xo-DaEjsxIma_AMhQkzGdM1rjS9AX2sBBKSir2CJ6qTT5SOsjynAflCILvjfCFWob4Mgf4W_GLlWdtmFKmqzdzEnm5kNpWDHfi3vuaZbDbYK0HXvsNW9s0ZS1FZypYJCnAS-Jwm0IgPxC5sVxjIVffCiL_R5Z0Vnm-D1NdB9Al4",
    "location":"Delhi",
    "server_ip": "164.52.216.211",
    "port": 22,
    "username": "root"
}
EOF
# preparing the e2e cloud node and deploying the image 
python3 utils/e2e_automation/start.py --automation-config="$automation_path/automation.json" --docker-image-path="$image_export_path"  --ssh-key-path=""$automation_path/ssh_keys"" \
                                      --ssh-keyname=$ssh_key_name --authorization-config="$automation_path/auth.json"