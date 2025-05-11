# The following script is used to automate the steps of creating an
# image from a simple sketch. The steps are:
# 1) DETR model: following model is used to fetch all the objects present in the sketch. (https://huggingface.co/docs/transformers/model_doc/yolos)
# 2) Preparing Prompt: following part is used to prepare a prompt using the above objects.
# 3) CLIP model: The following model is used to match the similarity between image and text. (https://huggingface.co/docs/transformers/model_doc/clip)
# 4) Canny-Edge Detection: following model is used to detect the edges of objects in the sketch.
# 4) ControlNET model: Following model is used to prepare segmentation map of the detection image.
# 5) StableDiffusion model: finally is the stable diffusion model which is used to prepare the image using segmentation map.

# loading path.sh file
. ./path.sh

# parameters
language="english"
pipeline=$1

if [ "$pipeline" = "sketch_to_video" ]; then
    # setting up data
    ./data.sh $language $pipeline
    # to train the model
    ./sketch_to_video/train/detr.sh 

else
    echo "No pipeline is provided"
    exit 1
fi
