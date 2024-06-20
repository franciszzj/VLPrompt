# VLPrompt: Vision-Language Prompting for Panoptic Scene Graph Generation

Official code implementation of VLPrompt, [arXiv](https://arxiv.org/abs/2311.16492).

## Abstract
Panoptic Scene Graph Generation (PSG) aims at achieving a comprehensive image understanding by simultaneously segmenting objects and predicting relations among objects.
However, the long-tail problem among relations leads to unsatisfactory results in real-world applications.
Prior methods predominantly rely on vision information or utilize limited language information, such as object or relation names, thereby overlooking the utility of language information. 
Leveraging the recent progress in Large Language Models (LLMs), we propose to use language information to assist relation prediction, particularly for rare relations.
To this end, we propose the Vision-Language Prompting (VLPrompt) model, which acquires vision information from images and language information from LLMs.
Then, through a prompter network based on attention mechanism, it achieves precise relation prediction.
Our extensive experiments show that VLPrompt significantly outperforms previous state-of-the-art methods on the PSG dataset, proving the effectiveness of incorporating language information and alleviating the long-tail problem of relations.

## Method
![hilo_overview](assets/method.png)
The overall framework of our VLPrompt method, which comprises three components: the vision feature extractor, the language feature extractor and the vision-language prompter.

## Results
![hilo_results](assets/results.png)
Comparison between our VLPrompt and other methods on the PSG dataset. Our method shows superior performance compared to all previous methods.

## Visualization
![visual_results](assets/vis_v1.png)
Visualization results of our VLPrompt.
We show two examples.
For each example, the top left displays the predicted segmentation results, the top right shows the top 10 predicted relation triplets (all are correct relation triplets), and bottom is the language snippet utilized for predicting the highlighted triplets in yellow.

## How to train
```
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
  --nproc_per_node=8 \
  --master_port=27500 \
  tools/train.py \
  $CONFIG \
  --auto-resume \
  --no-validate \
  --launcher pytorch
```

## How to test
Execute the following command, and you will get a submission file that can be used to evaluate the model.
```
PYTHONPATH=".":$PYTHONPATH \
python tools/infer.py \
  $EXP_TAG \
  $EPOCH_NUM
```

## How to evaluate
Please install [HiLo](https://github.com/franciszzj/HiLo).
```
cd ${HiLo_ROOT}
# SUBMISSION_PATH looks like "work_dirs/kings_sgg_v1_1/epoch_12_results/submission/"
python tools/grade.py $SUBMISSION_PATH
```

## Resource
Json used to train and test our method: [psg_train.json](https://emckclac-my.sharepoint.com/:u:/g/personal/k21163430_kcl_ac_uk/EUDvXDxSEexJnkBfy_1yr34BvJfimWQTUfOKEMTPwxyF0w?e=7tbF1R) and [psg_val.json](https://emckclac-my.sharepoint.com/:u:/g/personal/k21163430_kcl_ac_uk/Ecau5X4R8ylHsGc543BuqJsBggqhN8l3pLXT3-5TlVvzDg?e=5t1xVW).

## Citation
```
@article{zhou2023vlprompt,
  title={VLPrompt: Vision-Language Prompting for Panoptic Scene Graph Generation},
  author={Zhou, Zijian and Shi, Miaojing and Caesar, Holger},
  journal={arXiv preprint arXiv:2311.16492},
  year={2023}
}
```
