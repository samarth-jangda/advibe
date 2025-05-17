# The following script is used to train 
# the DETR model by carrying out following steps
# 1) class for repreparing json data with only image id and labels
# 2) function for training the model using training parameters
# 3) function for validating the model on validation dataset

import os
import gc
import json
import torch
import pickle
import scipy.io
import argparse
from tqdm import tqdm
import concurrent.futures
from datasets import Dataset
import multiprocessing as mp
from functools import partial
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import DetrForObjectDetection, DetrImageProcessor

parser = argparse.ArgumentParser(description="",
                                 epilog="",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--parameters-file-path", type=str, default="", required=True,
                    help="")

parser.add_argument("--train-tensor-json-path", type=str, default="", required=True,
                    help="")

parser.add_argument("--validation-tensor-json-path", type=str, default="", required=True,
                    help="")

parser.add_argument("--data-path", type=str, default="", required=True,
                    help="")

parser.add_argument("--train-output-path", type=str, default="", required=True,
                    help="")

args = parser.parse_args() 

# testing parameters
# train_tensor_json_path="/home/samarth-jangda/development/advibe/modules/data/module/general/detr/tensor_train_data.json"
# validation_tensor_json_path="/home/samarth-jangda/development/advibe/modules/data/module/general/detr/tensor_valid_data.json"
# data_path="/home/samarth-jangda/development/advibe/modules/data/module/general/detr/Scene/Sketch/paper_version"
# parameters_file_path="/home/samarth-jangda/development/advibe/modules/data/lib/detr/training_parameters.json"
# train_output_path="/home/samarth-jangda/development/advibe/modules/data/lib/detr"


def encode_data(dataset,processor,json_data_file):
    """
    The following function is to prepare the data in json
    with only required information which includes following
    {
        "image_id" : "unique id of each image",
        "labels" : "list of labels in image"
    }
    And, encode the data using pretrained DETR processor 
    """
    # load the matrix tensor file of image
    load_mat = scipy.io.loadmat(dataset["image_tensor_path"])
    image_tensor=torch.tensor(load_mat['normal_image'])
    if image_tensor.dtype != torch.uint8:
        image_tensor = (image_tensor * 255).clamp(0,255).to(torch.uint8)
    tensor_to_pil=T.ToPILImage()
    image_data=tensor_to_pil(image_tensor)
    annot_data=[annotation for annotation in json_data_file if annotation[0]['annotations']['image_id']==dataset['annotations']['image_id']][0]
    target_data={
        "image_id":dataset['annotations']['image_id'],
        "annotations":[data['annotations'] for data in annot_data]
    }
    image_encodings=processor(images=image_data,annotations=target_data,return_tensors="pt")
    encodings={
        k: v.squeeze(0) if isinstance(v, torch.Tensor) else v
        for k, v in image_encodings.items()
    }

    # Cleaning up memory variables
    del load_mat,image_tensor,image_data,tensor_to_pil,target_data,annot_data,image_encodings
    gc.collect()
    return encodings

def encoded_batches(batch,processor,json_data_file):
    """
    The following function is used to process encoding each batch
    using model processor and json annotations.
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        encoded_batch = list(executor.map(lambda sample: encode_data(sample, processor, json_data_file), batch))
    return {"encoded_data": encoded_batch}
    # encoded_batch = [encode_data(sample,processor,json_data_file) for sample in batch]
    # return {"encoded_data":encoded_batch}

def prepare_batches(data_type,dataset,batch_size,processor,json_data_file):
    """
    The following function is used to prepare batches and process it to
    model processor and write the output of each batch to disc.
    """
    # num_batches = len(dataset) // batch_size + (1 if len(dataset) % batch_size != 0 else 0)
    one_batch_data = len(dataset) // batch_size
    print(f"Data in one batch: {one_batch_data}")
    if not os.path.isdir(f"{args.train_output_path}/processing_batches/{data_type}"):
        os.mkdir(f"{args.train_output_path}/processing_batches/{data_type}")
    processed_output_path= f"{args.train_output_path}/processing_batches/{data_type}"
    for i in tqdm(range(batch_size),desc=f"Processing Batches of {data_type} dataset"):
        batch=dataset[i * one_batch_data: (i+1) * one_batch_data]
        encoded_batch=encoded_batches(batch,processor,json_data_file)
        with open(f"{processed_output_path}/processor_batch_{i}.pkl", 'wb') as batch_file:
            pickle.dump(encoded_batch,batch_file)
        del encoded_batch    
    return {"Encoding_Directory":processed_output_path,"Message":"Completed processing all batches","status_code":200}

# def prepare_batches(batch, batch_index, data_type, processor, json_data_file, train_output_path):
#     processed_output_path = f"{train_output_path}/processing_batches/{data_type}"
#     os.makedirs(processed_output_path, exist_ok=True)

#     print(f"[INFO] Processing batch {batch_index} of type {data_type}")
#     encoded_batch = encoded_batches(batch, processor, json_data_file)

#     with open(f"{processed_output_path}/processor_batch_{batch_index}.pkl", 'wb') as batch_file:
#         pickle.dump(encoded_batch, batch_file)
#     del encoded_batch
#     return f"Batch {batch_index} processed"

# def split_data(dataset,batch_size):
#     """
#     The following function is used to split the dataset into
#     given number of batches.
#     """
#     one_batch_data = len(dataset) // batch_size
#     print(f"Data in one batch: {one_batch_data}")
#     batches = [
#         dataset[i * one_batch_data : (i + 1) * one_batch_data]
#         for i in range(batch_size)
#     ]
#     return batches

# def run_batch(arg):
#     batch, index, function_to_run = arg
#     return function_to_run(batch, index)

# def multiprocess_encode(data_type,dataset,batch_size,processor,json_data_file):
#     """
#     The following function is used to process the data
#     on all cores of the machine to increase the processing speed.    
#     """
#     batches = split_data(dataset,batch_size)
#     function_to_run = partial(
#         prepare_batches,
#         data_type=data_type,
#         processor=processor,
#         json_data_file=json_data_file,
#         train_output_path=train_output_path
#     )
#     startmap_args = [(batch,i,function_to_run) for i,batch in enumerate(batches)]
#     results=[]
#     with mp.Pool(processes=min(mp.cpu_count(),batch_size)) as pool:
#         # tqdm wraps the iterator and updates as tasks complete
#         for result in tqdm(pool.imap_unordered(run_batch, startmap_args), total=len(startmap_args), desc="Processing Batches"):
#             results.append(result)
#     return results    

def check_batch_dtype(batch,device):
    """
    The following function is used to 
    check if the batch has data in tensor or list and 
    put it to device.
    """
    new_batch={}
    new_batch["labels"]=[]
    for k,v in batch.items():
        if isinstance(v,torch.Tensor):
            new_batch[k] = v.to(device)
        elif isinstance(v, list) and isinstance(v[0], torch.Tensor):
            new_batch[k] = [t[0].to(device) for t in v]
        elif k == "labels":
            for labels in v:
                new_batch["labels"].append({
                    "class_labels":labels["class_labels"].to(device),
                    "boxes": labels["boxes"].to(device),
                    "size": labels["size"].to(device),
                    "orig_size": labels["orig_size"].to(device),
                    "image_id": labels["image_id"].to(device)
                })
        else:
            new_batch[k] = v
    return new_batch                

def collate_data(data):
    """
    The following function is used to collate
    all the data present in dataset
    """
    batch_out = {
    "pixel_values": torch.stack([item['pixel_values'] for item in data]),
    "labels": []
    }

    for item in data:
        labels = item["labels"]
        batch_out["labels"].append({
            "class_labels": torch.cat([l["class_labels"] for l in labels]),
            "boxes": torch.cat([l["boxes"] for l in labels]),
            "size": labels[0]["size"],
            "orig_size": labels[0]["orig_size"],
            "image_id": labels[0]["image_id"]
        })
    max_class_label = 91  # max number of classes the model expects
    for label in batch_out['labels']:
        label['class_labels'] = label['class_labels'] % max_class_label
    return batch_out

def validate_model(model,validation_data_path,train_args,device):
    """
    The following function is used to validate the model of each batch on
    the validation set by providing the validation loss.
    """
    print("Starting Validation")
    for batch_files in tqdm(sorted(os.listdir(f"{validation_data_path}")),desc="Loading Batch Files"):
        with open(os.path.join(f"{validation_data_path}",batch_files), 'rb') as f:
            batch_file = pickle.load(f)
        validation_dataset = ModelDataset(batch_file['encoded_data'][:2])
        validation_dataloader = DataLoader(
            validation_dataset.samples,
            batch_size=train_args['training_batches'],
            shuffle=True,
            collate_fn=collate_data
        )
        model.eval()
        total_eval_loss = 0
        with torch.no_grad():
            for batch in validation_dataloader:
                batch = check_batch_dtype(batch,device)
                outputs = model(**batch)
                loss = outputs.loss
                total_eval_loss += loss.item()
        avg_val_loss = total_eval_loss/len(validation_dataloader) 
        print(f"Average Validation Loss: {avg_val_loss}")
        return avg_val_loss

def plot_model(batch_performance):
    """
    The following function is used to plot the results 
    of loss over time for each batch model and final model.
    """  

def train_model(device,train_data,validation_data_path,train_args,detr_model,optimizer,batch_filename):
    """
    The following function is used to train the model via
    following steps:
    1) Preparing the training arguments
    2) Carrying out the training loop
    """
    # scaler = GradScaler()
    batch_performance={
        "epochs":[],
        "loss":[]
    }
    processed_checkpoints=os.mkdir(f"{args.train_output_path}/checkpoints") if not os.path.isdir(f"{args.train_output_path}/checkpoints") else f"{args.train_output_path}/checkpoints"
    for epoch in tqdm(range(train_args["num_epochs"]),desc="Epochs"):
        detr_model.train()
        total_loss = 0
        for batch in train_data:
            optimizer.zero_grad()
            batch = check_batch_dtype(batch,device)
            # with autocast():
            outputs = detr_model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            # scaled_loss = scaler.scale(loss)
            # scaled_loss.backward()
            # scaler.step(optimizer)
            # scaler.update()
            total_loss += loss.item()
        batch_performance["epochs"].append(epoch)
        avg_loss = total_loss / len(train_data)    
        batch_performance["loss"].append(avg_loss)
    print(f"AverageLoss : {avg_loss}" )
    
    # code to save the model of each batch
    model_output = os.path.join(processed_checkpoints,f"detr_checkpoint_{batch_filename}.pth")
    torch.save(detr_model.state_dict(),model_output)

    # run the validation set
    print(f"Running validation on detr_checkpoint_{batch_filename}.pth")
    validation_results = validate_model(detr_model,validation_data_path,train_args,device)
    print(f"Validation Loss: {validation_results}")
    # time.sleep(5)
    del train_data
    gc.collect
    torch.cuda.empty_cache()
    # code to plot the model of each batch for accuracy over time period
    plot_batch = plot_model(batch_performance)

class ModelDataset:
    def __init__(self,samples):
        self.samples=samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,idx):
        sample=self.samples[idx]
         # If not tensor, convert to tensor
        processed_sample = {}
        for k, v in sample.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            processed_sample[k] = v

        return processed_sample  



def concatenate_model():
    """
    
    """

def combine_batch(dataset_path,train_batches,train_batch_dir):
    """
    The following function is used to combine multiple batches
    as per required training batches.
    """
    batch_index=0
    file=0
    # batch_dataset=[]
    total_files = os.listdir(dataset_path)
    per_train_batch_data = len(total_files) / train_batches
    # batch = {
    #     "encoded_data":[]
    # }
    os.mkdir(train_batch_dir)
    batch_out = open(f"{train_batch_dir}/train{batch_index}.pkl", 'ab')
    for i,batch_file in tqdm(enumerate(total_files),desc="Combining processed batches:"):
        with open(os.path.join(f"{dataset_path}",batch_file), 'rb') as f:
            batch_file = pickle.load(f)
        for item in batch_file['encoded_data']:
            pickle.dump(item,batch_out)    
        del batch_file    
        file += 1    
        gc.collect()
        # batch["encoded_data"].extend(batch_file['encoded_data']) 
        if i >= per_train_batch_data:
            batch_out.close()
            batch_index+=1
            file = 0
            batch_out = open(f"{train_batch_dir}/train{batch_index}.pkl", 'ab')
            per_train_batch_data += 5.0
    # return batch_dataset

def process_steps():
    """
    The following function is used to carry out following steps
    1) Encoding the train and validation dataset
    2) Training the DETR model
    3) Validating the DETR model
    """  
    # |----------------------------------------------------- Initializing data and model --------------------------------------------------|

    with open(args.parameters_file_path,'r') as training_arguments_file:
        train_args=json.load(training_arguments_file)
    
    # applying training parameters
    training_args={
        "num_batches":train_args['num_batches'],
        "training_batches":train_args["train_batches"],
        "dataloader_batches":train_args["dataloader_batches"],
        "num_epochs":train_args['num_epochs'],
        "learning_rate":train_args['learning_rate'],
        "weight_decay":train_args['weight_decay'],
        "save_dir":train_args['save_dir'],
        "log_interval":train_args['log_interval']
    }

    # initializing required parameters
    model_name="facebook/detr-resnet-50-dc5"
    model=DetrForObjectDetection.from_pretrained(model_name)
    processor=DetrImageProcessor.from_pretrained(model_name)
    device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=training_args["learning_rate"],weight_decay=training_args["weight_decay"])
    
    # fetch image path from training json file
    with open(args.train_tensor_json_path,'r') as train_tensor_file:
        train_json=json.load(train_tensor_file)
    
    flatten_train_data=[image_data for annotation in train_json for image_data in annotation][:12000]

    # fetch image path from validation json file
    with open(args.validation_tensor_json_path, 'r') as validation_tensor_file:
        validation_json = json.load(validation_tensor_file)
    
    flatten_validation_data=[image_data for annotation in validation_json for image_data in annotation][:3000]    
    
    # train and validation data encoding
    train_dataset=Dataset.from_list(flatten_train_data)
    validation_dataset=Dataset.from_list(flatten_validation_data)
    
    train_data=[]
    for row in train_dataset:
        filtered_data={
            "image_tensor_path": row["image_tensor_path"],
            "annotations": row["annotations"]
        }
        train_data.append(filtered_data)

    valid_data=[]
    for row in validation_dataset:
        filtered_data={
            "image_tensor_path": row["image_tensor_path"],
            "annotations": row["annotations"]
        }
        valid_data.append(filtered_data)

  # |----------------------------------------------------- Model Data Processing --------------------------------------------------|

    if not os.path.exists(f"{args.train_output_path}/processing_batches/process_batches"):
        encoded_train_data = prepare_batches("process_batches",train_data,training_args["num_batches"],processor,train_json)
    if not os.path.exists(f"{args.train_output_path}/processing_batches/valid"):
        encoded_valid_data = prepare_batches("valid",valid_data,training_args["num_batches"],processor,validation_json)
    # encoded_train_data = train_dataset.map(encode_data,fn_kwargs={"processor":processor,"json_data_file":train_json},remove_columns=["id","image_name","label_name"],num_proc=4)
    

    # combine pkl files to reduce number of batches
    if not os.path.exists(f"{args.train_output_path}/processing_batches/train"):
        train_batches = combine_batch(f"{args.train_output_path}/processing_batches/process_batches",training_args["training_batches"],f"{args.train_output_path}/processing_batches/train")

    # |----------------------------------------------------- model training --------------------------------------------------------|
    for batch_files in tqdm(sorted(os.listdir(f"{args.train_output_path}/processing_batches/train")),desc="Loading Batch Files"):
        with open(os.path.join(f"{args.train_output_path}/processing_batches/train",batch_files), 'rb') as f:
            batch_file = pickle.load(f)
        # print(batch_file)     
        train_dataset = ModelDataset([batch_file])
        train_dataloader = DataLoader(
            train_dataset.samples,
            batch_size=training_args["dataloader_batches"],
            shuffle=True,
            collate_fn=collate_data
        )
        batch_file_name = batch_files.split(".pkl")[0]
        validation_data_path = f"{args.train_output_path}/processing_batches/valid"
        train_detr = train_model(device,train_dataloader,validation_data_path,training_args,model,optimizer,batch_file_name)
        print("Completed with training DETR model")
        
if "__main__"==__name__:
    process_steps()    