# The following scrpt is used to push the image and corresponding container 
# to e2e cloud server and then start the container on server 

import argparse,os,time,paramiko,requests,json

parser=argparse.ArgumentParser(description="",
                               epilog="",
                               formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--automation-config", type=str, default="", required=True,
                    help="")

# args=parser.parse_args()
api_key="49e552ff-2a36-4fbc-866d-cb00626f6cd0"
project_id=33427
location="Delhi"
auth_token="eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJGSjg2R2NGM2pUYk5MT2NvNE52WmtVQ0lVbWZZQ3FvcXRPUWVNZmJoTmxFIn0.eyJleHAiOjE3NDc5Nzc4MTMsImlhdCI6MTcxNjQ0MTgxMywianRpIjoiMTliNTFmNDgtODJmNy00MTA4LWFhODctMWQ0OGYyZjAzNDZhIiwiaXNzIjoiaHR0cDovL2dhdGV3YXkuZTJlbmV0d29ya3MuY29tL2F1dGgvcmVhbG1zL2FwaW1hbiIsImF1ZCI6ImFjY291bnQiLCJzdWIiOiIyZDA2MTQwZi0zNDVhLTQ4YzktYTA3MC04NGFmZGYyNDRhNzYiLCJ0eXAiOiJCZWFyZXIiLCJhenAiOiJhcGltYW51aSIsInNlc3Npb25fc3RhdGUiOiJiMGQxYjFkMi05MjcwLTQ1ZGUtODI3Zi01YzhhNDE1ZmU4YWEiLCJhY3IiOiIxIiwiYWxsb3dlZC1vcmlnaW5zIjpbIiJdLCJyZWFsbV9hY2Nlc3MiOnsicm9sZXMiOlsib2ZmbGluZV9hY2Nlc3MiLCJ1bWFfYXV0aG9yaXphdGlvbiIsImFwaXVzZXIiLCJkZWZhdWx0LXJvbGVzLWFwaW1hbiJdfSwicmVzb3VyY2VfYWNjZXNzIjp7ImFjY291bnQiOnsicm9sZXMiOlsibWFuYWdlLWFjY291bnQiLCJtYW5hZ2UtYWNjb3VudC1saW5rcyIsInZpZXctcHJvZmlsZSJdfX0sInNjb3BlIjoicHJvZmlsZSBlbWFpbCIsInNpZCI6ImIwZDFiMWQyLTkyNzAtNDVkZS04MjdmLTVjOGE0MTVmZThhYSIsImVtYWlsX3ZlcmlmaWVkIjpmYWxzZSwibmFtZSI6IlNhbWFydGggSmFuZ2RhIiwicHJpbWFyeV9lbWFpbCI6InNpbmdodmluYXlkZXYzMjFAZ21haWwuY29tIiwiaXNfcHJpbWFyeV9jb250YWN0IjpmYWxzZSwicHJlZmVycmVkX3VzZXJuYW1lIjoic2FtYXJ0aC5qYW5nZGFAZ21haWwuY29tIiwiZ2l2ZW5fbmFtZSI6IlNhbWFydGgiLCJmYW1pbHlfbmFtZSI6IkphbmdkYSIsImVtYWlsIjoic2FtYXJ0aC5qYW5nZGFAZ21haWwuY29tIn0.Xo-DaEjsxIma_AMhQkzGdM1rjS9AX2sBBKSir2CJ6qTT5SOsjynAflCILvjfCFWob4Mgf4W_GLlWdtmFKmqzdzEnm5kNpWDHfi3vuaZbDbYK0HXvsNW9s0ZS1FZypYJCnAS-Jwm0IgPxC5sVxjIVffCiL_R5Z0Vnm-D1NdB9Al4"
def get_plans():
    """
    The following function is used to getch 
    The current plans in E2E Networks system
    display_category=GPU&category=GPU
    """
    base_url=f"https://api.e2enetworks.com/myaccount/api/v1/images/?display_category=GPU&category=GPU&osversion=22.04&gpu_type=&ng_container=null&apikey={api_key}&project_id={project_id}&location={location}"
    payload={}
    headers = {
    'Accept': 'application/json, text/plain, */*',
    'Authorization': f'Bearer {auth_token}',
    }
    response = requests.request("GET", base_url, headers=headers, data=payload)
    print(response.text)

def ssh_keys():
    api_key="49e552ff-2a36-4fbc-866d-cb00626f6cd0"
    project_id=33427
    url = f'https://api.e2enetworks.com/myaccount/api/v1/sshkeys/?apikey={api_key}&project_id={project_id}'
    response=requests.get(url)
    ssh_keys = response.json().get('data', [])
    
def start_server(server_id,username,ssh_key):
    """
    The following function is used to start the E2E 
    server after loading the image and container for specified time
    """
    base_e2e_url="https://api.e2enetworks.com/v2/servers/start"
    
    ssh=paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    connect_server=ssh.connect(hostname=server_id,port=22,username=username,key_filename=ssh_key)
    print("done")

def upload_image():
    """
    The following function is used to upload the build image to the 
    e2e server specified by the server_url 
    """


 
if "__main__" == __name__:
    # first will start the e2e server
    automation_config="/opt/SpeechEmotionRecognition/utils/e2e_automation/automation.conf"
    open_config_file=open(automation_config,'r').readlines()
    server_ip=open_config_file[0].split("=")[1].replace("\n","")
    server_port=open_config_file[1].split("=")[1].replace("\n","")
    username=open_config_file[2].split("=")[1].replace("\n","")
    image_name=open_config_file[3].split("=")[1].replace("\n","")
    ssh_key_path=open_config_file[4].split("=")[1].replace("\n","")
    ssh_keys()
    plans=get_plans()
    start=start_server(server_ip,username,ssh_key_path)