# The following script is used to call the openai gpt-4 model to 
# prepare a text prompt using the criterias.

import os
import json
import math
import openai
import argparse
from PIL import Image
from dotenv import load_dotenv
from text_prompt import (
    NUMBER_OF_OBJECTS,
    OBJECTS_LOCATION,
    OBJECT_LABELS,
    OBJECT_AREA,
    OBJECTS_RELATIONSHIP,
    CAMERA_ANGLE,
    CAMERA_ANGLE_FOR_TEXT
)

parser = argparse.ArgumentParser(description="",
                                 epilog="",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--generative-model-name", type="str", default="gpt-4o", required=False,
                    help="Please specify the name of the text generative model")

parser.add_argument("--detr-model-data", type=str, default="", required=True,
                    help="Please output the json list of output of detr model")

parser.add_argument("--sketch-image-data", type=str, default="", required=True,
                    help="Specify the path of testing set of sketch images")

parser.add_argument("--text-output", type="str", default="", required=True,
                    help="Please specify the path to store generated text")

args = parser.parse_args()

load_dotenv()
openai.api_key=os.getenv("OPENAI_KEY")

#arguments for testing  

def run_model(prompt):
    """
    The following function is used to call the generative
    model and prompt in order to generate required response.
    """
    model_name = args.generative_model_name
    model_response = openai.ChatCompletion.create(
        model = model_name,
        message = prompt,
        temperature = 0.0
    )
    return model_response


class ObjectsMetaData:
    """
    
    """
    def __init__(self,image_id,detr_model_data):
        self.detected_image_id = image_id
        self.detected_data = detr_model_data
        

    def object_count(self):
        """
        The following function is used to count the
        number of objects as detected by object detection model
        """
        total_objects = len(self.detected_data[self.detected_image_id]["object_id"])
        return total_objects

    def object_location(self):
        """
        The following function is used to find the
        coordinates of intersection oi dialgonals of rectangle
        formed by the bounding box in the sketch image.
        Use the midpoint formulae:
        ((x1+x2)/2 , (y1+y2)/2)
        """
        intersections = []
        for boxes in self.detected_data[self.detected_image_id]["bbox"]:
            line_1 = [boxes[0],boxes[2]] # (x0,y0) and (x2,y2)
            mid_x = (line_1[0][0] + line_1[1][0]) / 2
            mid_y = (line_1[0][1] + line_1[1][1]) / 2
            intersection_point = (mid_x,mid_y)
            intersections.append(intersection_point)
        return intersections

    def object_area(self):
        """
        The following function is used to find the area of rectangle
        formed by the bounding box of object in the sketch image.
        applying distance formulae to find length and bredth.
        dictanc_formulate = sqroot( sq(x2-x1) + sq(y2-y1) ) 
        """
        object_areas=[]
        for boxes in self.detected_data[self.detected_image_id]["bbox"]:
            number_1 = boxes[0][0] - boxes[1][0]
            number_2 = boxes[0][1] - boxes[1][1]
            length_1 = math.sqrt(math.pow(number_1,2) + math.pow(number_2,2))
            number_3 = boxes[0][0] - boxes[3][0]
            number_4 = boxes[0][1] - boxes[3][1]
            length_2 = math.sqrt(math.pow(number_3,2) + math.pow(number_4,2))
            object_area = length_1 * length_2
            object_areas.append(object_area)
        return object_areas    

class ObjectRelationship:
    """
    
    """
    def __init__(self,model_name,prompt_list):
        gen_model = model_name
        prompts = prompt_list

    def load_sketch_image():
        """
        The following funcftion is used to load the sketch image
        """    

    def find_camera_angle():
        """
        The following function uses generative modeel 
        by uploadin the sketch image and the camera angle prompt in
        order to find whether the skecth image is of close shot or long shot. 
        """

    def object_relationship():
        """
        The following function also uses generative model by
        uploading the sketch image and the object relationship prompt
        in order to find the relationship of each object.
        """


def image_text_descriptiopn():
    """
    The following function is used to prepare the final
    text description using the:
    1) Objects Meta Data of each skecth image
    2) Object Relationship and Image Shot Angle.
    """
    messages = [
        {
            "role": "system",
            "content":(
                "You are a helpful assistant whose task is to analyze the uploaded image"
                "and describe the same on the basis or number of objects, location, camera_angle and relationship" 
            )
        },
        {
            "role": "user",
            "content":(
                f"""
                You are a helpful assistant whose task is to analyze the pencil sketch on the basis of
                following criterias
                1) Number of Objects: The total number of objects which are present in a prepared pencil sketch are {NUMBER_OF_OBJECTS}
                2) Object Labels: The labels which are present in the pencil sketch are {OBJECT_LABELS}
                2) Camera Angle: {CAMERA_ANGLE}
                3) Location of Objects: {OBJECTS_LOCATION}
                4) Relationship in objects: {OBJECTS_RELATIONSHIP}
                Output the prepared text in not more than 60 words in json format with following key
                Image_Prompt: Generated Text
                NOTE: Do not assume any thing on you own, strictly follow the provied criterias 
            """
            )
        }
    ]

def process_data():
    """
    The following function is used to load all images of test set
    and generate final text description of each of the image, prepare
    a json file for each imagetext description and save in output file.

    1) To load each test image.
    2) To fetch all the metadata from detr model output file
    3) Then find the objects relationship and the image angle.
    4) Finally pass all the metadata, objects relationship and image angle
        to text generation model and get the final response.
    5) Save the generated text description of each image in a json file. 
    """ 
    text_gen_model = args.generative_model_name
    for image_name in args.sketch_image_data:
        load_image = Image.open(f"{args.sketch_image_data}/{image_name}")
        with open(args.detr_image_data, 'r') as detr_data_file:
            load_detr_data = json.load(detr_data_file)
        detected_image_data = [detr_data for detr_data in detr_data_file if image_name == detr_data["image_id"].split("-")[1]]
        if detected_image_data !=[]:
            image_id = list(detected_image_data.keys())[0]
            metadata = {
                image_id:{
                    "labels": detected_image_data[0][image_id]["image_labels"]
                }
            }
            object_metadata_class = ObjectsMetaData(image_id,detected_image_data[0])
            metadata[image_id]["object_count"] = object_metadata_class.object_count()
            metadata[image_id]["objct_locations"] = object_metadata_class.object_location()
            metadata[image_id]["object_areas"] = object_metadata_class.object_area()


if "__main__" == __name__:
    process_data()
