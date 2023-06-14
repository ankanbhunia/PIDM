# Person Image Synthesis via Denoising Diffusion Model [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ankanbhunia/PIDM/blob/main/PIDM_demo.ipynb)

 <p align='center'>
  <b>
    <a href="https://arxiv.org/abs/2211.12500">ArXiv</a>
    | 
    <a href="https://ankanbhunia.github.io/PIDM">Project</a>
    | 
    <a href="https://colab.research.google.com/github/ankanbhunia/PIDM/blob/main/PIDM_demo.ipynb">Demo</a>
    |
    <a href="https://www.youtube.com/watch?v=cHdZTZurX8M">Youtube</a>
  </b>
</p> 
<p align="center">
<img src=Figures/images.gif>
 
## News

- **2023.02** A demo available through Google Colab:

    :rocket:
    [Demo on Colab](https://colab.research.google.com/github/ankanbhunia/PIDM/blob/main/PIDM_demo.ipynb)
    


 
## Generated Results

<img src="https://raw.githubusercontent.com/ankanbhunia/PIDM/main/Figures/intro_fig.jpg"> 

You can directly download our test results from Google Drive: (1) [PIDM.zip](https://drive.google.com/file/d/1zcyTF37UrOmUqtRwwq1kgkyxnNX3oaQN/view?usp=share_link) (2) [PIDM_vs_Others.zip](https://drive.google.com/file/d/1iu75RVQBjR-TbB4ZQUns1oalzYZdNqGS/view?usp=share_link)

The [PIDM_vs_Others.zip](https://drive.google.com/file/d/1iu75RVQBjR-TbB4ZQUns1oalzYZdNqGS/view?usp=share_link) file compares our method with several state-of-the-art methods e.g. ADGAN [14], PISE [24], GFLA [20], DPTN [25], CASD [29],
NTED [19]. Each row contains target_pose, source_image, ground_truth, ADGAN, PISE, GFLA, DPTN, CASD, NTED, and PIDM (ours) respectively. 




## Dataset

- Download `img_highres.zip` of the DeepFashion Dataset from [In-shop Clothes Retrieval Benchmark](https://drive.google.com/drive/folders/0B7EVK8r0v71pYkd5TzBiclMzR00). 

- Unzip `img_highres.zip`. You will need to ask for password from the [dataset maintainers](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html). Then rename the obtained folder as **img** and put it under the `./dataset/deepfashion` directory. 

- We split the train/test set following [GFLA](https://github.com/RenYurui/Global-Flow-Local-Attention). Several images with significant occlusions are removed from the training set. Download the train/test pairs and the keypoints `pose.zip` extracted with [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) by downloading the following files:

<!--   ```bash
  cd scripts
  ./download_dataset.sh
  ```

  Or you can download these files manually： -->

  - Download the train/test pairs from [Google Drive](https://drive.google.com/drive/folders/1PhnaFNg9zxMZM-ccJAzLIt2iqWFRzXSw?usp=sharing) including **train_pairs.txt**, **test_pairs.txt**, **train.lst**, **test.lst**. Put these files under the  `./dataset/deepfashion` directory. 
  - Download the keypoints `pose.rar` extracted with Openpose from [Google Driven](https://drive.google.com/file/d/1waNzq-deGBKATXMU9JzMDWdGsF4YkcW_/view?usp=sharing). Unzip and put the obtained floder under the  `./dataset/deepfashion` directory.

- Run the following code to save images to lmdb dataset.

  ```bash
  python data/prepare_data.py \
  --root ./dataset/deepfashion \
  --out ./dataset/deepfashion
  ```
## Custom Dataset

The folder structure of any custom dataset should be as follows:

- dataset/
- - <dataset_name>/
- - - img/ 
- - - pose/
- - - train_pairs.txt
- - - test_pairs.txt

You basically will have all your images inside ```img``` folder. You can use different subfolders to store your images or put all your images inside the ```img``` folder as well. The corresponding poses are stored inside ```pose``` folder (as txt file if you use openpose. In our project, we use 18-point keypoint estimation). ```train_pairs.txt``` and ```test_pairs.txt``` will have paths of all possible pairs seperated by comma ```<src_path1>,<tgt_path1>```.

After that, run the following command to process the data:

```
python data/prepare_data.py \
--root ./dataset/<dataset_name> \
--out ./dataset/<dataset_name>
--sizes ((256,256),)
```

This will create an lmdb dataset ```./dataset/<dataset_name>/256-256/```




## Conda Installation

``` bash
# 1. Create a conda virtual environment.
conda create -n PIDM python=3.6
conda activate PIDM
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# 2. Clone the Repo and Install dependencies
git clone https://github.com/ankanbhunia/PIDM
pip install -r requirements.txt

```
## Method

<img src=Figures/main.png>

## Training 

This code supports multi-GPUs training.

  ```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port 48949 train.py \
--dataset_path "./dataset/deepfashion" --batch_size 8 --exp_name "pidm_deepfashion"

  ```


## Inference 

Download the pretrained model from [here](https://drive.google.com/file/d/1WkV5Pn-_fBdiZlvVHHx_S97YESBkx4lD/view?usp=share_link) and place it in the ```checkpoints``` folder.
For pose control use ```obj.predict_pose``` as in the following code snippets. 

  ```python
from predict import Predictor
obj = Predictor()

obj.predict_pose(image=<PATH_OF_SOURCE_IMAGE>, sample_algorithm='ddim', num_poses=4, nsteps=50)

  ```

For apperance control use ```obj.predict_appearance```

  ```python
from predict import Predictor
obj = Predictor()

src = <PATH_OF_SOURCE_IMAGE>
ref_img = <PATH_OF_REF_IMAGE>
ref_mask = <PATH_OF_REF_MASK>
ref_pose = <PATH_OF_REF_POSE>

obj.predict_appearance(image=src, ref_img = ref_img, ref_mask = ref_mask, ref_pose = ref_pose, sample_algorithm = 'ddim',  nsteps = 50)

  ```

The output will be saved as ```output.png``` filename.


## Citation

If you use the results and code for your research, please cite our paper:

```
@article{bhunia2022pidm,
  title={Person Image Synthesis via Denoising Diffusion Model},
  author={Bhunia, Ankan Kumar and Khan, Salman and Cholakkal, Hisham and Anwer, Rao Muhammad and Laaksonen, Jorma and Shah, Mubarak and Khan, Fahad Shahbaz},
  journal={CVPR},
  year={2023}
}
```

[Ankan Kumar Bhunia](https://scholar.google.com/citations?user=2leAc3AAAAAJ&hl=en),
[Salman Khan](https://scholar.google.com/citations?user=M59O9lkAAAAJ&hl=en),
[Hisham Cholakkal](https://scholar.google.com/citations?user=bZ3YBRcAAAAJ&hl=en), 
[Rao Anwer](https://scholar.google.fi/citations?user=_KlvMVoAAAAJ&hl=en),
[Jorma Laaksonen](https://scholar.google.com/citations?user=qQP6WXIAAAAJ&hl=en),
[Mubarak Shah](https://scholar.google.com/citations?user=p8gsO3gAAAAJ&hl=en) &
[Fahad Khan](https://scholar.google.ch/citations?user=zvaeYnUAAAAJ&hl=en&oi=ao)
 
