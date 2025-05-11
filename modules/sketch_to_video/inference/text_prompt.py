NUMBER_OF_OBJECTS="""
The number of objects are which are detected in a sketch image
The following data will be useful in order to understand the depth,
background scene and the relationsip between each object in the sketch image

The total number of objects which are present in a prepared pencil sketch are {objects_number}
"""

OBJECT_LABELS="""
The image labels are the names of the detected objects in the sketch image
Use the following data of list of labels in order to understand the foundation
of scene of the image and the relationship along with context between all labels
The list of detected labels are {detected_labels}
"""

OBJECT_AREA="""
In the image the objects covers some area which here is the
object area in square pizels. Use the following data in order to 
understand the structure of the image and hence relationship between 
each obejct.
The metadata of each object and its area list is :
{object_area_metadata}
"""

# a different prompt in order to find the camera angle 
CAMERA_ANGLE="""
The camera angle tells whether the image present is a long shot or a 
close shot. To understand this better consider some examples.

Examples:
a) Close Shot: A sketch which contains very few objects, like only a human, a animal.
b) Long Shot: A sketch which containes lot of objects like a sketch with sky,trees and animals.

The total number of objects present in the sketch image are {objects_count}
"""

CAMERA_ANGLE_FOR_TEXT="""
The camera angle tells whether a sketch image is of 
long shot or close shot. Both scenarios matter in their own way in order to 
understand the scene of the sketch image.
Now if camera angle is of close shot then it means the skecth is highly focused
on given detected objects but if the camera angle is of long shot then it means
the image focuses on capturing more details of the scene. 

Understand the following carefully in order to understand the meaning of
an image.

The type of shot of sketch image: {camera_angle_shot}
"""

# The objects location metadata is 
# metadata = {
#     "object_label" : {
#         "area": area in integer,
#         "location" : coordinates of object
#     }
# }

OBJECTS_LOCATION = """
The object location is the coordinate of the intersection of diagonals of
bounding boxes of detected objects in the sketch image. Now, use the data of
location of each detected object along with each object's area in order to understand
where each object is placed and what is its role in the image. 
The metadata of area and location of each detected object is {objects_location_metadata} 
"""

OBJECTS_RELATIONSHIP="""
Define the importance (definition, how to use it)
"""

IMAGE_GEN_TEXT_PROMPT=f"""
    You are a helpful assistant whose task is to analyze the pencil sketch on the basis of
    following criterias
    1) Number of Objects: {NUMBER_OF_OBJECTS}
    2) Camera Angle: {CAMERA_ANGLE}
    3) Location of Objects: {OBJECTS_LOCATION}
    4) Relationship in objects: {OBJECTS_RELATIONSHIP}
    Output the prepared text in not more than 60 words in json format with following key
    Image_Prompt: Generated Text
    NOTE: Do not assume any thing on you own, strictly follow the provied criterias 
"""