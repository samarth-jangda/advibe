# The following script is used to carry out all steps for DETR model which includes
# 1) Training DETR model
# 2) Inference of DETR model
# 3) Improving training hyperparameters based on current model performance
    # Compare confidence scores, IoU, mAP, false positives/negatives.
    # Adjust learning rate, batch size, dropout, training epochs.

# calling the path script
. ./path.sh


[ ! -d "$output_model_path/detr" ] && echo "Preparing the output model directory for detr model" && mkdir "$output_model_path/detr"
[ ! -d "$output_model_path/detr/checkpoints" ] && echo "Preparing checkpoints directory for detr model" && mkdir "$output_model_path/detr/checkpoints"
[ ! -d "$output_model_path/detr/processing_batches" ] && echo "Preparing batches directory to save processed data of train and validation" && mkdir "$output_model_path/detr/processing_batches"
[ ! -d "$output_model_path/detr/validation" ] && echo "Preparing directory for saving validation results" && mkdir "$output_model_path/detr/validation"


# setting up the training arguments for detr model
checkpoint_dir="$output_model_path/detr/checkpoints"
cat <<EOF > "$output_model_path/detr/training_parameters.json"
{
"num_batches": 1,
"train_batches": 10,
"num_epochs": 25,
"learning_rate": 1e-5,
"weight_decay": 0.01,
"save_dir": "$checkpoint_dir",
"log_interval": 10
}

EOF

python3 sketch_to_video/train/train_detr.py --parameters-file-path="$output_model_path/detr/training_parameters.json" --train-tensor-json-path="$detr_path/tensor_train_data.json" \
                      --data-path="$general/detr/Scene/Sketch/paper_version" --validation-tensor-json-path="$detr_path/tensor_valid_data.json" \
                      --train-output-path="$output_model_path/detr"

# inference all models on testing dataset

models_count=$(find "$output_model_path/detr/checkpoints" -type f | wc -l)
echo "Running test data on $models_count number of detr models"
[ -d "$output_model_path/detr/results" ] && echo "Preparing directory for saving results of inference through all models" && mkdir "$output_model_path/detr/results"

python3 sketch_to_video/inference/detect_object.py --testing-data="" --models-directory="$output_model_path/detr/checkpoints" --output-path="$output_model_path/detr/results"