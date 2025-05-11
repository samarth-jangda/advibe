#!/bin/bash
# The following script is used to set all directories and 
# downloading all the required data in the respective directories.
# NOTE: User can give the path of a zip file with all domain data to be used by system
# dataset command : gdown https://drive.google.com/uc?id=1ApjDhGjtqfFEMzm6dmyhS-2aXnnYLxnj 
# and https://github.com/nightrome/cocostuff to download stuff_trainval2017.zip file

# calling the paths script
. ./path.sh

# setting required variables
language=$1
pipeline=$2
domain_file_path=$3
detr_data="$models_data/Scene.tar" 


# checking required parameters
if [ -z "$language" ] || [ -z "$pipeline" ] ; then
    echo "Error: No language or pipeline is provided"
    echo "Usage: ./data.sh <language> <pipeline> <domain_file_path>"
    exit 1
fi

if [ -z "$domain_file_path" ]; then
    echo "Warning: No domain file is provided"
    echo "Usage: ./data.sh <language> <pipeline> <domain_file_path>"
    echo "Continuing without any domain file"
fi


# building parent directories
[ ! -d "$module_data" ] && mkdir "$module_data"
[ ! -d "$general" ] && mkdir "$general"
[ ! -d "$domain" ] && mkdir "$domain"
[ ! -d "$output_model_path" ] && mkdir "$output_model_path"

if [ $pipeline == "sketch_to_video" ]; then
    # preparing all data directories
    [ ! -d "$detr_path" ] && echo "Preparing data directory for detection transformer" && mkdir "$detr_path"
    [ ! -d "$stable_diffusion_path" ] && echo "Preparing data directory for stable diffusion" && mkdir "$stable_diffusion_path"
    [ ! -d "$prompts_data_path" ] && echo "Preparing data directory for saving prompt files" && mkdir "$prompts_data_path"
    
    # preparing training , testing, validation directories
    for dir in "$detr_path" "$stable_diffusion_path"; do
        [ ! -d "$dir/domain" ] && echo "Preparing domain data directory in $dir" && echo "$dir/domain"
        [ ! -d "$dir/train" ] && echo "Preparing training directory in $dir" && echo "$dir/train"
        [ ! -d "$dir/test" ] && echo "Preparing testing directory in $dir" && echo "$dir/test"
        [ ! -d "$dir/validate" ] && echo "Preparing validate directory in $dir" && echo "$dir/validate"
    done

    # downloading data in required directories
    current_dir=$(pwd)
    if [ ! -d "$current_dir/$python_env_name" ]; then
        echo "Preparing python environment in : $current_dir"
        python3 -m venv "$python_env_name"
        source "$current_dir/$python_env_name/bin/activate"
        pip3 install -r "$current_dir/requirements.txt"
    else
        source "$current_dir/$python_env_name/bin/activate"    
    fi

    # to download sketch images (COCOSketchy) data for detection transformer
    
    # setting general dataset
    echo "File Path: $detr_path/val2017.zip"

    if [ ! -d "$detr_path/val2017" ]; then
        gdown --id 1U2g-RXo9ua45gxeSbgiAiaI8imoOi6fO -O "$detr_path"
        tar -xvf "$detr_path/Scene.tar" -C "$detr_path"
        wget -P "$detr_path" "http://images.cocodataset.org/zips/val2017.zip"
        wget -P "$detr_path" "http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuff_trainval2017.zip"
        unzip "$detr_path/val2017.zip" -d "$detr_path"
        unzip "$detr_path/stuff_trainval2017.zip" -d "$detr_path"
    fi
    # setting domain dataset
    if file "$domain_file_path" | grep -q 'Zip archive data'; then
        [ ! "$domain_file_path" == "None" ] && cp -r "$domain_file_path" "$domain"
    else
        echo "$domain_file_path is not a zip file."
    fi
    

    # data validation and formatting
    # example format
    cat <<EOF > "$data_path/data_format.json"
{
    
    "images": [
        {
        "id": "data id",
        "file_name": "image name",
        "width": "image-width",
        "height": "image-height"
        }
    ],

    "annotations": [
        {
        "id": "data id",
        "image_id": "id of image",
        "category_id": "detected category id",
        "bbox": "coordinates of bounding box",
        "area": "arear by box in int",
        "segmentation": "the segmentation coordinates",
        "iscrowd": 0
        }
    ],

    "categories": [
        {"id": "data id", "name": "category name"},
        {"id": "data id", "name": "category name"},
        {"id": "data id", "name": "category name"}
    ]
    
}
EOF

fi
    [ ! -d "$module_data/image_tensor" ] && echo "Preparing image tensor directory for saving train and validation data" && mkdir "$module_data/image_tensor"
    # starting with data processing
    python3 sketch_to_video/train/data_processing.py --processing-data-path="$module_data" --general-image-path="$detr_path/Scene/Sketch/paper_version" --domain-image-path="$domain" \
                                               --stuff-json-path="$detr_path/stuff_train2017.json" --sample-format="$data_path/data_format.json" --image-size="800-700" --model-path=""
    

