# The following script is used to do these tasks:
# 1) Fetch Plans: Fetch the corresponding plans of nodes in E2E Networks cloud.
# 2) Create node: Creating node in E2E Networks cloud.
# 3) Upload Image: Uploading created image on either created node or e2e registry.

import paramiko.client,docker
import argparse,json,requests,paramiko,os


parser=argparse.ArgumentParser(description="",
                               epilog="",
                               formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--automation-config", type=str, default="", required=True,
                    help="Specify the configuration file of automation")

parser.add_argument("--docker-image-path", type=str, default="", required=True,
                    help="Specify the path of docker image")

parser.add_argument("--ssh-key-path", type=str, default="", required=True,
                    help="Specify the path where to save the generated ssh key")

parser.add_argument("--ssh-keyname", type=str, default="", required=True,
                    help="Specify the ssh keyname by which it will be generated")

parser.add_argument("--authorization-config", type=str, default="", required=True,
                    help="Speciy the authorization configuration")


# args=parser.parse_args()

class CreateNode:
    def __init__(self,os,gpu_require,delete,price,cpu_price,cpu_specs,gpu_specs,auth_file_path):
        self.operating_system=os #---------------------------------------> The operating system of the node
        self.gpu_required=gpu_require #----------------------------------> Boolean field to check gpu requirement
        self.delete_node=delete #----------------------------------------> Boolean field to check to delete node
        self.node_price=price #------------------------------------------> The price of the node
        self.cpu_price=cpu_price #---------------------------------------> The proce of the node if gpu is not used
        self.specs_list=cpu_specs #--------------------------------------> The specification list without gpu 
        self.gpu_specs_list=gpu_specs #----------------------------------> The specification list with gpu
        self.auth_file=auth_file_path #----------------------------------> The path or json data of authorization file
        self.nodes_data=None #-------------------------------------------> Store the data for present nodes
        
    def fetch_plans(self):
        """
        The following function 
        1) Will fetch the plans of machines as per arguments
        2) Prepare a math to select an appropriate machine
        """
        node_price=[]
        node_specs=[]
        api_key=self.auth_file.get("api_key")
        project_id=self.auth_file.get("project_id")
        location=self.auth_file.get("location")
        auth_token=self.auth_file.get("auth_token")
        if self.gpu_required==True:
            nodes_data=f"https://api.e2enetworks.com/myaccount/api/v1/images/?display_category=GPU&category=GPU&osversion=22.04&gpu_type=&ng_container=null&apikey={api_key}&project_id={project_id}&location={location}"
        else:
            nodes_data=f"https://api.e2enetworks.com/myaccount/api/v1/images/?category=Ubuntu&osversion=22.04&gpu_type=&ng_container=null&apikey={api_key}&project_id={project_id}&location={location}"
        headers = {
        'Accept': 'application/json, text/plain, */*',
        'Authorization': f'Bearer {auth_token}',
        }
        response = requests.request("GET", nodes_data, headers=headers, data={})
        nodes_data=json.loads(response.text)
        for data in nodes_data.get("data"):
            if "GPU" in data.get('image'):
                # process as per gpu machine
                # NOTE: Fow simplicity there is no logic written for selecting machine both on price and specs
                node_name_price=data if data.get('specs').get('price_per_hour') > self.node_price[0] and data.get('specs').get('price_per_hour') > self.node_price[0] else print("High price")
                node_name_specs=data if data.get('specs').get('ram') == float(self.gpu_specs_list.get("ram")) and data.get('specs').get("disk_space") == self.gpu_specs_list.get("disk_space") and data.get('specs').get("cpu") < self.gpu_specs_list.get("cpu") else print("Not required")
            else:
                node_name_price=data if data.get('specs').get('price_per_hour') <= float(self.cpu_price) else print("High price")
                node_name_specs=data if float(data.get('specs').get('ram')) == float(self.specs_list.get('ram')) and data.get('specs').get('disk_space') == self.specs_list.get("disk_space") else print("Not required") 
            node_price.append(node_name_price)
            node_specs.append(node_name_specs)
        return {"Node_Data_Price":node_price[0],"Node_Data_Specs":node_specs[0]}
    
    def fetch_sshkeys(self):
        """
        The following function is used to fetch the sshkeys from
        the E2E Networks server to connect with the new node
        """
        api_key=self.auth_file.get("api_key")
        project_id=self.auth_file.get("project_id")
        url = f'https://api.e2enetworks.com/myaccount/api/v1/sshkeys/?apikey={api_key}&project_id={project_id}'
        response=requests.get(url)
        ssh_keys = response.json().get('data', [])
        return {"SSH_Key":ssh_keys}
        
    def create_node(self,node_price_data,node_specs_data):
        """
        The following function is used to:
        1) Create the node in the e2e server as per specs from
        fecth_plans function
        """
        if self.gpu_required==True:
            data=node_price_data
        else:
            data=node_specs_data
        node_label="Default"
        region=self.auth_file.get('location')   
        api_key=self.auth_file.get("api_key")
        project_id=self.auth_file.get("project_id")
        auth_token=self.auth_file.get("auth_token")
        base_api=f"https://api.e2enetworks.com/myaccount/api/v1/nodes/?apikey={api_key}&project_id={project_id}"
        headers = {
            'x-api-key': api_key, 
            'Content-Type': 'application/json', 
            'Authorization' : f'Bearer {auth_token}' 
        }
        payload = json.dumps({
            "label": f"{node_label}",
            "name": f"{data.get("name")}",
            "region": f"{region}",
            "plan":f"{data.get("plan")}",
            "image": f"{data.get('image')}",
            "ssh_keys": [],
            "backups": False,
            "enable_bitninja": False,
            "disable_password": False,
            "is_saved_image": False,
            "saved_image_template_id": None,
            "reserve_ip": "",
            "is_ipv6_availed": False,
            "vpc_id": "",
            "default_public_ip": False,
            "ngc_container_id": None,
            "security_group_id": 3482,
            "start_scripts": []
              })
        response=requests.request("POST",base_api,data=payload,headers=headers)
        print(response.status_code)
        node_resp=response.content
        return {"Node_Response":node_resp}

def generate_sshkeys(ssh_key_path,ssh_key_name,auth_file):
    """
    The following function is used to: 
    1) generate the ssh keys from backend to be used in created node
    2) upload the ssh key in the e2e cloud network
    """
    api_key=auth_file.get("api_key")
    project_id=auth_file.get("project_id")
    auth_token=auth_file.get("auth_token")
    if len(os.listdir(ssh_key_path)) != 0:
        print("Reading the existing ssh key")
    else:    
        key=paramiko.RSAKey.generate(bits=2048)
        with open(f"{ssh_key_path}/{ssh_key_name}", 'w') as private_filename:
            key.write_private_key(private_filename)
        with open(f"{ssh_key_path}/{ssh_key_name}.pub", 'w') as public_key_filename:
            public_key_filename.write(f"{key.get_name()} {key.get_base64()}\n")
    read_keyfile=open(f"{ssh_key_path}/{ssh_key_name}.pub", 'r').read().strip()    
    # upload_url = f'https://api.e2enetworks.com/myaccount/api/v1/sshkeys/?apikey={api_key}&project_id={project_id}'
    # headers = {
    #         'x-api-key': api_key, 
    #         'Content-Type': 'application/json', 
    #         'Authorization' : f'Bearer {auth_token}' 
    # }
    # upload_key={
    #     'label': "SamarthSSHKey",
    #     'public_key':read_keyfile
    # }
    # response=requests.post(upload_url,upload_key,headers=headers)
    # if response.status_code==200:
    #     ssh_key_id=response.json().get(id)
    #     return {"SSH_id":ssh_key_id,"filepath":f"{ssh_key_path}/{ssh_key_name}"}
    # else:
    #     return{"Message":"Unable to generate ssh id","PriavateKey":f"{ssh_key_path}/{ssh_key_name}"}
    return {"KeyPath":f"{ssh_key_path}/{ssh_key_name}.pub"}
    


def fetch_node_details(auth_data):
    """
    The following function is used to fetch the details
    of the created node
    """
    # node_id=node_response['id']
    node_id="164.52.217.72"
    api_key=auth_data.get("api_key")
    auth_token=auth_data.get("auth_token")
    node_url=f"https://api.e2enetworks.com/myaccount/api/v1/nodes/{node_id}/?apikey={api_key}"
    headers = {
            'x-api-key': api_key, 
            'Content-Type': 'application/json', 
            'Authorization' : f'Bearer {auth_token}' 
    }  
    current_node_response=requests.get(node_url,headers=headers)
    node_detail=current_node_response.json()
    hostname=node_detail.get('data')[0]["public_ip_address"]
    username="root"
    return {"Hostname":hostname,"Username":username}

def upload_image_to_registry(image_name,registry_url):
    """
    The following function is used to:
    1) Upload the docker image to e2e networks registry
    2) Connect the docker image to created node in e2e and start process
    """
    client= docker.from_env()
    tagged_image_name=f"{registry_url}/{image_name}"
    client.login(username='samarth.jangda@gmail.com', password='u0Cf6Vd3Mn6Zz8Xs2Z', registry='registry.e2enetworks.net')
    client.images.push(tagged_image_name)
    return {"TaggedImage":tagged_image_name}
    
def deploy_image_to_server(tagged_image,image_path,server_ip):
    
    target_client=docker.DockerClient(base_url=f"tcp://{server_ip}:22")
    with open(image_path, 'rb') as image_file:
        target_client.images.load(image_file.read())
    # Optionally, run containers using the loaded image on the target server
    container = target_client.containers.run(f'{tagged_image}', detach=True)
    return {"Message": "Run Successful"}

def process_pipeline(automation_config,auth_config,docker_image_path):
    """
    The following function is used to:
    1) Process the configurations to all functions and automate the process
    """
    # validating the config file
    # open_json=json.load(automation_config)
    with open(automation_config, 'r') as automation_file:
        load_config=json.load(automation_file)
    with open(auth_config, 'r') as authorization_file:
        load_auth_config=json.load(authorization_file)
    set_parameters=CreateNode(load_config.get("operating_system"),load_config.get("gpu_required"),load_config.get("delete_node"),
                              [load_config.get("min_node_price"),load_config.get("max_node_price")],load_config.get("cpu_node_price"),
                              load_config.get("cpu_specs_list"),load_config.get("gpu_specs_list"),load_auth_config)
    fetch_all_plans=set_parameters.fetch_plans()
    fetch_sshkeys=set_parameters.fetch_sshkeys()
    sshkey=generate_sshkeys(ssh_key_path,ssh_keyname,load_auth_config) if len(fetch_sshkeys.get("SSH_Key")) == 0 else fetch_sshkeys[0]['id'] # ----> showing Unauthoriuzed
    # create_node=set_parameters.create_node(fetch_all_plans.get("Node_Data_Price"),fetch_all_plans.get("Node_Data_Specs"))
    # fetch_node_data= fetch_node_details(load_auth_config)
    new_ssh_key="/opt/SpeechEmotionRecognition/utils/e2e_automation/ssh_keys/id_rsa_new"
    image_name=load_config.get("image_name")
    image_to_registry=upload_image_to_registry(image_name,f"{load_auth_config.get("registry_url")}/{load_auth_config.get("project_id")}")
    load_image_on_node=deploy_image_to_server(image_to_registry.get("TaggedImage"),docker_image_path,"164.52.217.72")
    print("done")
    
if "__main__" == __name__:
    ssh_key_path="/opt/SpeechEmotionRecognition/utils/e2e_automation/ssh_keys"
    ssh_keyname="id_rsa"
    automation_config="/opt/SpeechEmotionRecognition/utils/e2e_automation/automation.json"
    authorization_config="/opt/SpeechEmotionRecognition/utils/e2e_automation/auth.json"
    docker_image_path="/opt/SpeechEmotionRecognition/utils/e2e_automation/model.tar"
    process_pipeline(automation_config,authorization_config,docker_image_path)
    