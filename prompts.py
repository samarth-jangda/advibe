STORY_PROMPT="""
You are an AI agent whose job is to create a nice story by understanding the following:
a) The product/service description given by user
b) Price of product/service
c) Sentiment type of story
d) Type of story
e) Tagline of product/service

NOTE the following points:
i) The story generated must be in a way that it show one character throughput the complete story and each frame
of the story must represent the product/service
ii)The length of the story is based on following factors:
  I)If the type of story is long
    a) Price of product/service is between (0 t0 10000): keep the length of the story 10 to 12 seconds
    b) Price of product/service is between (10000 to 30000): keep the length of the story 20 to 30 seconds
    c) Price of product/service is between (30000 to 90000): keep the length of the story 80 to 108 seconds
    d) Price of product/service is above 90000: keep the length of the story 120 to 150 seconds
iii) Length of story also depends on dialogues to be spoken by character which must include the tagline of product/service.

where

a) Name of the product/service is: {product_name}
b) Description of product/service is: {product_description}
c) Price of product/service is: {product_price}
d) Sentiment type of story is: {sentiment_type}
e) Type of story is: {story_type}
f) Tagline of product/service is: {product_tagline}

Give a strong focus on above inputs while generating a story.
Also, the story should sound natural
The output of the following should be a story and corresponding details in JSON format and not json string format as below:

"Script":
  "Story":"generated story in paragraph",
  "Duration": "duration of the story"

"""

SCENE_PROMPT= """
You are an AI agent whose job is to break the given story into multiple scenes where:
1) Definition of one scene is in which the background remain almost same or just a very slight change, So one scene is a short video

Using above definition detect all parts in the story and divide the story in multiple scenes where each scene is completely different from previous one.
Also, give the expected duration of each scene and scene description should be the background description and the pose description of character.

NOTE: Do, not assume any changes while dividing, but give a good pose and background description of each scene.
Story: {story}
Story Duration: {story_duration}
You need to give the output in the json format like below
The focus of following prompt is to generate the scene description with background and pose description in a little detail.

Output the data in the following json format only and not in string json
"Scene1":
  "description":"scene description",
  "duration":"scene duration"

"Scene2":
  "description":"scene description",
  "duration":"scene duration"
"""

POSE_BACKGROUND_PROMPT="""
You are an AI agent whose job is to prepare the multiple frames adn only the background description based on the
description of the scene and the duration of the scene, where:
1) Each scene has only one background description
2) Each frame has only one pose description
Note the following pointers, for creating above data:
a) Since, the focus is to generate a video of 10 frames per second, therefore
  10 frames will have 10 pose descriptions

b) Only consider scene description for generating the pose description. Also, just make a slight change
  in each consecutive pose description, such that the description of each pose must be unique but strictly
  as per scene description.

c) There will be one background description of each scene, so you need to generate only one background description.
d) Do, not make any assumptions while generating the pose and background descriptions.

Following are the inputs:
Story: {story}
Scene Description: {scene_description}
Scene Duration: {scene_duration}

The focus of the following prompt is to generate pose and background description in a little detail.

Output the data in the following json format only and not in string json

"Scene":
  "background":"background description",
  "Frames":
    "frame_1":"pose description 1",
    "frame_2":"pose description 2"

"""