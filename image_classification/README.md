# lpcvc-cls
Solution for LPCVC classification track.

## Prerequisites

* TF12 environment
    - TensorFlow version = 1.12
    - Pytorch version >= 1.0
    - Python version >= 3.6
    - Nvidia GPUs
    
* TF15 environment
    - Tensorflow version = 1.15
    - Python version >= 3.6
    - Nvidia GPUs
    
 * Download [checkpoint](https://drive.google.com/open?id=1X9jmAM4mvBZ8BZZ6LKEI_S93qxkiTIHS) and put it into **root directory** (./finetune)
  

## Convert pth to tfinit with 1001 classes
Run the following command: 
``````bash
SPLIT=all # dataset split [train, all]
LATENCY=33 # latency of the model
bash get_tfinit.sh $SPLIT $LATENCY
``````
You will get the tfinit file under the resource constraint 33ms trained on train+val dataset.
## Get the ckpt for the model
Run the following command: 
``````bash
GPU=0 # the GPU id
SPLIT=all # dataset split [train, all]
LATENCY=33 # latency of the model
IMAGE_SIZE=192 # image size of the model input
bash train_proxy.sh $GPU $SPLIT $LATENCY $IMAGE_SIZE
``````
You will get the ckpt file under the resource constraint 33ms trained on train+val dataset.
## Freeze the ckpt and get tflite model
Run the following command: 
``````bash
GPU=0
SPLIT=all
LATENCY=33
IMAGE_SIZE=192
bash post_quant.sh $GPU $SPLIT $LATENCY $IMAGE_SIZE
``````
You will get the tflite model under the resource constraint 33ms trained on train+val dataset.

## Test the performance of tflite
Downlaod [imagenet_accuracy_eval](https://drive.google.com/file/d/1nHg0V_zPw_7kHr1dVbyYrC8J4UIjC5Ju/view?usp=sharing), which is a binary file we prebuilt for evaluation.

Run the following commands: 
``````bash
# Push the binary to android phone
adb push lpcvc/imagenet_accuracy_eval /data/local/tmp 

# Make the binary file executable
adb shell chmod +x /data/local/tmp/imagenet_accuracy_eval 

# Push you model to Android phone
adb push {NAME_OF_MODEL}.tflite /data/local/tmp

# Directly evalutation on Android phone
adb shell /data/local/tmp/imagenet_accuracy_eval \
  --model_file=/data/local/tmp/{NAME_OF_MODEL}.tflite \
  --ground_truth_images_path=/data/local/tmp/ilsvrc_images \
  --ground_truth_labels=/data/local/tmp/ilsvrc_validation_labels.txt \
  --model_output_labels=/data/local/tmp/model_output_labels.txt \
  --output_file_path=/data/local/tmp/accuracy_output.txt \
  --num_images=0
``````
You will do the evaluation on ImageNet validation dataset on your Android Phone.