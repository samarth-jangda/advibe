# The script is used to carry out folow9ng tasks
# 1) Validate and Create data in COCO format
# 2) Preprocess the image dataset ahnd convert to image tensor

import os
import uuid
import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy.io import savemat
from torchvision import transforms


parser=argparse.ArgumentParser(description="",
                               epilog="",
                               formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--processing-data-path", type=str, default="", required=True,
                    help="")

parser.add_argument("--general-image-path", type=str, default="", required=True,
                    help="")

parser.add_argument("--domain-image-path", type=str, default="", required=False,
                    help="")

parser.add_argument("--stuff-json-path", type=str, default="", required=True,
                    help="")

parser.add_argument("--sample-format", type=str, default="", required=False,
                    help="")

parser.add_argument("--image-size", type=str, default="800X400", required=True,
                    help="")

parser.add_argument("--model-path", type=str, default="Nil", required=False,
                    help="")

args=parser.parse_args()

# testing parameters
# processing_data_path="/home/samarth/development/advibe/modules/data/module"
# general_image_path="/home/samarth/development/advibe/modules/data/module/general/detr/Scene/Sketch/paper_version"
# domain_image_path=""
# stuff_json_path="/home/samarth/development/advibe/modules/data/module/general/detr/stuff_train2017.json"
# sample_format="/home/samarth/development/advibe/modules/data/data_format.json"
# image_size="800-700"
# model_path=""

# --------------------------------------COCO format validation-----------------------------------------#

class DataFormatValidation:
    """
    The following class is used to carry out following tasks:
    1) To validatet if the data is present in COCO format
    2) If data is not present in COCO format then use the pipeline to reprepare the data in
        COCO format 
    """
    def __init__(self,data):
        self.dataset=data
        

    def coco_format_validation():
        """
        The following function is used to check if the data
        is present in coco format
        """

# --------------------------------------Data Preprocessing---------------------------------------------#

class DataPreprocessing:
    """
    The following data processing class is used to carry out following tasks:
    1) Image Resize: The detection model requires image of size 800X400, therefore preparing all images of same size
    2) Image Normalize: To convert pizel values between [0,1] using mean and std dev 
    3) Image Tensor: Since DETR model expects a tensor , hence preparing tensor of each image
    4) Image Padding: If the image size is smaller than required then image will be padded with zeroes at edges.
    """
    def __init__(self,data,coco_data,image_size,tensor_path):
        self.image = data
        self.coco_json_file=coco_data
        self.image_size = int(image_size.split("-")[0]),int(image_size.split("-")[1])
        self.tensor_path = tensor_path
        self.padded_image=None # padded image to required size
        self.normal_image=None # normalized image tensor
        self.tensor_json_data=[]
    
    def image_padding(self):
        """
        If the size of the image is smaller than dimension of 800X400 then
        it is important to resize the image to match the required size with 
        zeroes at the edges.
        """
        # to check if the size of each image in the dataset 
        # is equals to required size
        image=cv2.imread(self.image)
        x,y=image.shape[:2] 
        a,b=self.image_size   # required size of the image
        if (x,y) < (a/0.5,b/0.5) or (x,y) > (a/10,b/10):
            # calculating padding from all sides
            top = (a-x)//2
            bottom = a-x-top
            left=(b-y)//2
            right=b-y-left
            padding_image=cv2.copyMakeBorder(image,top,bottom,left,right,cv2.BORDER_CONSTANT,value=(0,0,0))
            # Convert to grayscale
            gray_image = cv2.cvtColor(padding_image, cv2.COLOR_BGR2GRAY)
            self.padded_image=np.array(Image.fromarray(gray_image))
        else:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            self.padded_image=np.array(Image.fromarray(gray_image)) 
               
    
    def image_normalize(self):
        """
        The following function is used to resize the image
        to 800X600 , since DETR model only expects image of
        the following size.
        
        Normalizing an image is important for better understanding and
        computation, hence normalizing each image between 0 and 1 using mean
        and standard deviation
        """
        
        re_image=Image.fromarray(self.padded_image)
        image_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ])
        normalized_image=image_transform(re_image.convert("RGB"))
        self.normal_image=normalized_image
        
        
    def save_image_tensor(self,image_name):
        """
        The following function is used to save the image tensor
        into given tensor path
        """
        # save the tensor as mat file
        savemat(f"{self.tensor_path}/{image_name}.png.mat", {"normal_image": self.normal_image.cpu().numpy()})
        image_name=image_name.replace(".png",".mat")
        return {"TensorPath":f"{self.tensor_path}/{image_name}"}
    
    def tensor_json(self,image,tensor_path):
        """
        The following function is used to prepare a tensor json
        mapping file with following attributes
        NOTE: The image extension in COCO Stuff json is jpg not png
        JSON FILE
        {
            bbox:[],
            segmentation_mask:[],
            image_tensor_path:"",
            labels:[],
            label_tensor:[]
        }
        """
        image_name=image.replace(".png",".jpg")
        
        img_data=[json_data for json_data in self.coco_json_file["images"] if json_data["file_name"]==image_name][0]
        if img_data!=[]:
            segmentation_data=[data for data in self.coco_json_file['annotations'] if data['image_id']==img_data['id']]
            for segments in segmentation_data:
                tensor_json={
                    "id":None,
                    "image_name":None,
                    "image_tensor_path":None,
                    "annotations":None,
                    "label_name":None
                }
                tensor_json["id"]=str(uuid.uuid4())
                tensor_json["image_name"]=image
                tensor_json["image_tensor_path"]=tensor_path
                tensor_json["annotations"]=segments
                self.tensor_json_data.append(tensor_json)
                
        return {"TensorJsonData":self.tensor_json_data}    
        
def data_split():
    """
    The following function is used to divide the data in train,
    test and validate , where
    train will be 70% data
    test will be 20% data
    validate will be 10% data
    """
    

def process_dataset():
    """
    
    """
    image_data={
        "train":{
            "tensor_json_file_path": "/home/samarth-jangda/development/advibe/modules/data/module/general/detr/tensor_train_data.json",
            "data_json_file_path": f"{args.stuff_json_path}",
            "images": f"{args.general_image_path}/trainInTrain"
            
        },
        "validation":{
            "tensor_json_file_path": "/home/samarth-jangda/development/advibe/modules/data/module/general/detr/tensor_valid_data.json",
            "data_json_file_path": "/home/samarth-jangda/development/advibe/modules/data/module/general/detr/stuff_val2017.json",
            "images": f"{args.general_image_path}/val"  
        }
    }
    
    for key in list(image_data.keys()):
        tensor_json=[]
        with open(image_data[key]["data_json_file_path"], 'r') as caption_file:
            load_caption_file=json.load(caption_file)
        
        # create data_outputs directory
        print(f"Preparing directory for {key} tensor data")
        if not os.path.exists(f"{args.processing_data_path}/image_tensor/{key}")==True:
            os.mkdir(f"{args.processing_data_path}/image_tensor/{key}")
        
        for image in tqdm(os.listdir(image_data[key]["images"]),desc=f"Processing {key} dataset"):
            if not image.startswith("."):
                image_dir_path=image_data[key]["images"]
                initialize_processing_class=DataPreprocessing(f"{image_dir_path}/{image}",load_caption_file,args.image_size,f"{args.processing_data_path}/image_tensor/{key}")
                pad_image=initialize_processing_class.image_padding()
                resize_normalize=initialize_processing_class.image_normalize()
                save_image_tensor=initialize_processing_class.save_image_tensor(image)
                prepare_tensor_json=initialize_processing_class.tensor_json(image,save_image_tensor["TensorPath"])
                tensor_json.append(prepare_tensor_json["TensorJsonData"])
            
            
        # writing json to tensor json path
        tensor_path=image_data[key]["tensor_json_file_path"]
        with open(tensor_path, 'w') as tensor_json_file:
            json.dump(tensor_json,tensor_json_file,indent=4)

if "__main__"==__name__: 
    process_dataset()
