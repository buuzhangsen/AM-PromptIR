# Adaptive Multi-modal Prompting for All-in-One Image Restoration Amidst Diverse Degradations
The above content contains all the code details of the "Adaptive Multi-modal Prompting for All-in-One Image Restoration Amidst Diverse Degradations". As the article has not yet been published, we will only provide all our visual output results.This paper is currently submitted to  ```The Vision Computer ```.

<hr />
> **Abstract:** *The field of image restoration aims to recover high-quality images from their degraded counterparts, encompassing tasks such as denoising, deraining, and dehazing. In real-world scenarios, images often suffer from complex and diverse degradations, posing significant challenges for existing restoration models. To address this, we propose AM-PromptIR, an adaptive multi-modal image restoration framework that leverages implicit relationships between diverse degradation types to guide the restoration process. Specifically, we propose an Adaptive Multi-modal Prompt Generation (AMPG) module that leverages a pre-trained language-image model to capture the implicit relationships between various degradation types. By refining these relationships and dynamically adjusting the attention distribution of image features, AMPG enhances the model's understanding of the intricate interplay between images and their degradation patterns. Additionally, we introduce a Simplified Mixed Attention Mechanism (SMAM) and a Feature Enhancement and Fusion Module (FEFM) to improve feature extraction and fusion, respectively. The experimental results showed that compared with the baseline model PromptIr, AM-PromptIR improved PSNR by 9.2% in dehazing  tasks, 5.3% in deraining operations, and an average PSNR increase of 3.0% in various degraded tasks.These findings highlight the robustness and effectiveness of our approach in handling diverse image degradations.* 
<hr />

## Network Architecture

<img src = "AM-PromptIR.jpg"> 
The above diagram shows the overall architecture of the AM PromptIR network, and the corresponding code is mainly stored in the  ```net/main.py ``` file.


## Dataset Download and Preperation

All the 5 datasets used in the paper can be downloaded from the following locations:

Denoising: [BSD400](https://drive.google.com/file/d/1idKFDkAHJGAFDn1OyXZxsTbOSBx9GS8N/view?usp=sharing), [WED](https://drive.google.com/file/d/19_mCE_GXfmE5yYsm-HEzuZQqmwMjPpJr/view?usp=sharing), [Urban100](https://drive.google.com/drive/folders/1B3DJGQKB6eNdwuQIhdskA64qUuVKLZ9u)

Deraining: [Train100L&Rain100L](https://drive.google.com/drive/folders/1-_Tw-LHJF4vh8fpogKgZx1EQ9MhsJI_f?usp=sharing)

Dehazing: [RESIDE](https://sites.google.com/view/reside-dehaze-datasets/reside-v0) 

The training data should be placed in ``` data/Train/{task_name}``` directory where ```task_name``` can be Denoise,Derain or Dehaze.
After placing the training data the directory structure would be as follows:
```
└───Train
    ├───Dehaze
    │   ├───high
    │   └───low
    ├───Denoise
    └───Derain
        ├───high
        └───low
```

The testing data should be placed in the ```test``` directory wherein each task has a seperate directory. The test directory after setup:

```
├───dehaze
│   └───SOTS
│       ├───low
│       └───high
├───denoise
│   ├───bsd68
│   └───urban100
└───derain
    └───Rain100L
        ├───low
        └───high
```

## Environmental installation

All experimental environment dependency packages in this paper are located in the 'indispensable.txt' file, and the installation command is:
```
pip install indispensable.txt
```
## Training

After preparing the training data in ```data/``` directory, use 
```
python train.py
```
to start the training of the model. Use the ```de_type``` argument to choose the combination of degradation types to train on. By default it is set to all the 3 degradation types (noise, rain, and haze).

Example Usage: If we only want to train on deraining :
```
python train.py --de_type derain
```

## Testing

After preparing the testing data in ```test/``` directory, place the mode checkpoint file in the ```ckpt``` directory.  To perform the evalaution use
```
python test.py --mode {n}
```
```n``` is a number that can be used to set the tasks to be evaluated on, 0 for denoising, 1 for deraining, 2 for dehaazing and 3 for all-in-one setting.

Example Usage: To test on all the degradation types at once, run:

```
python test.py --mode 3
```

## Results
Performance results of the AM-PromptIR framework trained under the all-in-one setting,The visualization results of this experiment can be downloaded from [Output](https://drive.google.com/file/d/1q5cxl8Bs00m5aVHe2tIbe--ErX6QSjz-/view?usp=sharing).

<summary><strong>Table</strong> </summary>
The following table is a comparison chart of experimental evaluation indicators.
<img src = "result.png"> 

<summary><strong>Visual Results</strong></summary>
The following is a visual comparison of image dehazing.
<img src = "dehazy.jpg"> 
The following is a visual comparison of image derain.
<img src = "derain.jpg"> 



## Contact
**Acknowledgment:** This code is based on the [AirNet](https://github.com/XLearning-SCU/2022-CVPR-AirNet) and [PromptIR](https://github.com/va1shn9v/PromptIR) repositories. 

