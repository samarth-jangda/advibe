# The following file uses DETR model to detect all objects in the sketch
# The output of the following file will be a json with keys as object number and
# value as object names along with position and bounding box height & length
# OUTPUT as follows:
# image_id: {
#     "image_name": "The name of the image"
#     "object_id": "The unique id of the detected bject"
#     "labels": "The labels(names) of each object"
#     "bbox": "The coordinate of bbox of each object"
# }
# NOTE: image_id should be uuid-image_name

# imort necessary modules
import os
import torch
import logging
import argparse
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image, ImageFont, ImageDraw
from transformers import DetrForObjectDetection, DetrImageProcessor

parser=argparse.ArgumentParser(description="The following file is used to detect all objects in the given image",
                               epilog="",
                               formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--testing-data", type=str, default="", required=False,
                    help="Following argument is used to detect objects in testing data")

parser.add_argument("--model-directory", type=str, default="", required=False,
                    help="Please specify the input image path")

parser.add_argument("--output-path", type=str, default="", required=True,
                    help="Please specify the output path to save json")

args = parser.parse_args()


# testing arguments
# testing_data="/home/samarth-jangda/development/advibe/modules/data/inference/input"
# output_path="/home/samarth-jangda/development/advibe/modules/data/inference/output"
# model_directory=""

if not args.model_directory:
    model_name = "facebook/detr-resnet-50-dc5"
else:
    model_name = "final_checkpoint"   
model = DetrForObjectDetection.from_pretrained(model_name)
model_processor = DetrImageProcessor.from_pretrained(model_name)


model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for image in os.listdir(args.testing_data):

    # Process the image to object detection processor.
    image_data = Image.open(f"{args.testing_data}/{image}").convert("RGB")
    process_image = model_processor(image_data, return_tensors = "pt").to(device)
    process_image = {k: v.to(device) for k, v in process_image.items()}

    # Start forward pass to model
    with torch.no_grad():
        output = model(**process_image)

    # Post process predictions
    target_sizes = torch.tensor([image_data.size[::1]]).to(device)
    results = model_processor.post_process_object_detection(output,threshold=0.9,target_sizes=target_sizes)
    logging.info(f"Object Detection Results: {results}")

    # Visualize results
    draw_image = image_data.copy()
    draw_image = draw_image.copy().convert("RGB")
    draw = ImageDraw.Draw(draw_image)
    image_font = ImageFont.load_default()
    for score,label,bbox in zip(results[0]["scores"],results[0]['labels'],results[0]['boxes']):
        image_name=image.split(".")[0]
        box = [round(i,2) for i in bbox.tolist()]
        draw.rectangle(box,outline="red",width=2)
        draw.text((box[0], box[1]), f"{model.config.id2label[label.item()]}: {round(score.item(), 2)}", fill="red", font=image_font)
        draw_image.save(f"{args.output_path}/{image_name}_out.jpeg") 

# save the results in a json file