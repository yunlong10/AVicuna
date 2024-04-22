# AVicuna
Repo for the paper ["AVicuna: Audio-Visual LLM with Interleaver and Context-Boundary Alignment for Temporal Referential Dialogue"](https://arxiv.org/abs/2403.16276).

---

## Installation

We recommend setting up a conda environment for the project:
```shell
conda env create -f avicuna.yml
conda activate avicuna

git clone https://github.com/yunlong10/AVicuna.git
cd AVicuna
```

## Data
Download the meta data in JSON [here]() and place them into the `./data` folder.

```
- data
    - stage1.json
    - stage2.json
    - stage3.json
    - stage4.json
```

The video and audio features can be extracted by `./avicuna/get_clip.py` and `./avicuna/get_clap.py`. You can also down the extracted features [here]().

## Training
We train our model on a single NVIDIA A6000 48G GPU.

Stage I: Vision-Text Alignment
```shell
bash scripts/stage1.sh
```

Stage II: Audio-Text Alignment
```shell
bash scripts/stage2.sh
```

Stage III: Context-Boundary Alignment
```shell
bash scripts/stage3.sh
```

Stage IV: Instruction Tuning
```shell
bash scripts/stage4.sh
```

## Checkpoints
Download the fine-tuned checkpoints

## Inference

```python
python -m avicuna.inference
```

## Acknowledgements

We are grateful for the following awesome projects our AVicuna arising from:

* [LLaVA](https://github.com/haotian-liu/LLaVA): Large Language and Vision Assistant
* [FastChat](https://github.com/lm-sys/FastChat): An Open Platform for Training, Serving, and Evaluating Large Language Model based Chatbots
* [Video-ChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT): Towards Detailed Video Understanding via Large Vision and Language Models
* [LLaMA](https://github.com/facebookresearch/llama): Open and Efficient Foundation Language Models
* [Vid2seq](https://github.com/google-research/scenic/tree/main/scenic/projects/vid2seq): Large-Scale Pretraining of a Visual Language Model for Dense Video Captioning
* [VTimeLLM](https://github.com/huangb23/VTimeLLM): A Vid-LLM for Fine-grained Video Moment Understanding
* [InternVid](https://github.com/OpenGVLab/InternVideo/tree/main/Data/InternVid): A Large-scale Video-Text dataset
* [UnAV-100](https://unav100.github.io): Untrimmed Audio-Visual Video dataset


## Citation

```bibtex
@article{tang2024avicuna,
  title={AVicuna: Audio-Visual LLM with Interleaver and Context-Boundary Alignment for Temporal Referential Dialogue},
  author={Tang, Yunlong and Shimada, Daiki and Bi, Jing and Xu, Chenliang},
  journal={arXiv preprint arXiv:2403.16276},
  year={2024}
}
```


