# The following script is used to inference through all the models involved
# is sketch to video pipeline which includes
# 1) Object Detection : The following part is used to output the objects information in input sketch image
# 2) Text Prompt : The following part is used to preapre a text prompt using defined prompt criterias.
# 3) Edge Detection : The following part is used to detect all the edges in the input sketch image.
# 4) Text Encoder : The following part is used to encode the text prompt.
# 5) Image Encoder : The following part is used to encode the canny edge sketch image.
# 6) Image Generation : Following is the final part which produces final realistic image as input sketch image.
# NOTE: It is expected to have checkpoints directory in every model directory

# calling the paths script
. ../path.sh

language=$1

# checking if all model directories have checkpoints directories
models_list=("detr" "text_encoder" "image_encoder" "image_gen")
models=()

for model in "${models_list[@]}"; do
    [ ! -d "$output_model_path/$model/checkpoints" ] && echo "Warning: No directory of checkpoints exists in $model, hence using base model"
    if [ ! -z "$(ls -A "$output_model_path/$model/checkpoints")" ]; then 
        final_model=$(ls -t "$output_model_path/$model/checkpoints" | head -n 1)
        models+=($output_model_path/$model/checkpoints/$final_model)
    else
        echo "Warning: checkpoints directory is empty in $model, hence using base model"
    fi    
done

detr_model_name="${models[0]}"
echo "Running inference through Object detection model: $detr_model_name"
if [ ! -z "$detr_model_name" ]; then
    python3 inference/detect_object.py --testing-data="$input_path" --model-directory="$detr_model_name" --output-path="$output_path"
else
    echo "Running inference through base model"
    python3 inference/detect_object.py --testing-data="$input_path" --output-path="$output_path"
fi    