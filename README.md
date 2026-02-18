### SAM3-LiteText: An Anatomical Study of the SAM3 Text Encoder for Efficient Vision-Language Segmentation

[Chengxi Simon Zeng](https://simonzeng7108.github.io/about/)<sup>1,†</sup>, [Yuxuan Jiang](https://YuxuanJJ.github.io/)<sup>1</sup>, [Gao Ge](https://scholar.google.com/citations?user=j2_80ewAAAAJ&hl=en)<sup>1</sup>, [Shuai Wang](https://shuaiwang97.github.io/)<sup>2</sup>, [Duolikun Danier](https://scholar.google.com/citations?user=Example)<sup>3</sup>, [Bin Zhu](https://scholar.google.com/citations?user=Example)<sup>4</sup>, [Stevan Rudinac](https://scholar.google.com/citations?user=Example)<sup>2</sup>, [David Bull](https://scholar.google.com/citations?user=Example)<sup>1</sup>, [Fan Aaron Zhang](https://fan-aaron-zhang.github.io/)<sup>1</sup>
<sup>1</sup>Visual Information Lab, University of Bristol; <sup>2</sup>MultiX lab, University of Amsterdam; <sup>3</sup>University of Edinburgh; <sup>4</sup>Singapore Management University

<sup>†</sup>Tech Lead & Corresponding Author

[📄 Paper](https://arxiv.org/abs/2602.12173) | [🌐 Project Page](https://simonzeng7108.github.io/efficientsam3/) | [🤗 Hugging Face](https://huggingface.co/Simon7108528/EfficientSAM3) | [💬 Discord](https://discord.gg/FMyaQca7xT)



## Table of Contents

- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Inference](#inference)
- [Training and Evaluation](#training-and-evaluation)
- [Datasets](#datasets)
- [SAM3-LiteText Model Zoo \& Weight Release](#sam3-litetext-model-zoo--weight-release)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Users](#users)

---

[SAM3](https://github.com/facebookresearch/sam3) (Segment Anything Model 3) enables flexible, prompt-driven visual grounding, but inherits large, general-purpose text encoders originally designed for open-ended language understanding. In practice, segmentation prompts are short, structured, and semantically constrained (e.g., `white dog`, `person in blue jacket`), leading to substantial over-provisioning in text encoder capacity and persistent computational and memory overhead.

In this paper, we perform a large-scale anatomical analysis of text prompting in vision–language segmentation, covering 404,796 real prompts across multiple benchmarks. Our analysis reveals severe redundancy: most context windows are underutilized, vocabulary usage is highly sparse, and text embeddings lie on a low-dimensional manifold despite high-dimensional representations.

Motivated by these findings, we propose **SAM3-LiteText**, a lightweight text encoding framework that replaces the original SAM3 text encoder with a compact MobileCLIP student that is optimized by knowledge distillation. Extensive experiments on image and video segmentation benchmarks show that SAM3-LiteText reduces text encoder parameters by up to 88%, substantially reducing static memory footprint, while maintaining segmentation performance comparable to the original model.





---

## Installation

EfficientSAM3 purposely shares the same software contract as upstream SAM3:

- **Python** ≥ 3.12
- **PyTorch** 2.7.0 (CUDA 12.6 build recommended)
- **CUDA**-capable GPUs with drivers that support CUDA ≥ 12.6

Follow the exact environment setup from the [official SAM3 README](sam3/README.md) or use the condensed steps below (single-node example):
EfficientSAM3 purposely shares the same software contract as upstream SAM3:

- **Python** ≥ 3.12
- **PyTorch** 2.7.0 (CUDA 12.6 build recommended)
- **CUDA**-capable GPUs with drivers that support CUDA ≥ 12.6

Follow the exact environment setup from the [official SAM3 README](sam3/README.md) or use the condensed steps below (single-node example):

```bash
git clone https://github.com/SimonZeng7108/efficientsam3.git
cd efficientsam3

conda create -n efficientsam3 python=3.12 -y
conda activate efficientsam3

pip install --upgrade pip
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install repo dependencies via the root pyproject (brings in SAM3 + Stage-1 extras)
pip install -e ".[stage1]"
```

---

## Inference

Download checkpoints from the [Model Zoo](#efficientsam3-model-zoo--weight-release) section. All Stage 1 image encoder weights are available via Google Drive and Hugging Face links in the table below.

**Quick Start:**

<p align="center">
  <img src="https://github.com/SimonZeng7108/efficientsam3/blob/main/images/es-tv-mc-m-teaser.png" width="30%">
</p>

<p align="center">
  <img src="https://github.com/SimonZeng7108/efficientsam3/blob/main/images/dog_person_example_dog.png" width="45%">
  <img src="https://github.com/SimonZeng7108/efficientsam3/blob/main/images/dog_person_example_person.png" width="45%">
</p>

 **MobileCLIP-S1 (63.56M)** distilled from **SAM3 Text Encoder (353.72M)**

```python
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from PIL import Image
import torch

# 1. Build SAM3-LiteText model (single call handles text encoder swap + checkpoint + truncation)
model = build_sam3_image_model(
    checkpoint_path="output/efficient_sam3_image_encoder_mobileclip_s1_ctx16.pt",
    load_from_HF=False,
    text_encoder_type="MobileCLIP-S1",      # "MobileCLIP-S0", "MobileCLIP-S1", "MobileCLIP2-L"
    text_encoder_context_length=16,          # 16, 32, or 77
)

# 2. Inference
processor = Sam3Processor(model, confidence_threshold=0.4)
image = Image.open("image.jpg").convert("RGB")
state = processor.set_image(image)
state = processor.set_text_prompt("dog", state)
masks = state["masks"]
scores = state["scores"]
```
---

## Training and Evaluation

**Training:**
- For SAM3 Text encoder distillation training details, see [README_stage1.md](README_stage1.md).


---

## Datasets

For dataset setup and download scripts (`data/download_*.sh`) covering COCO, DAVIS, LVIS, SA-1B, SA-V, LVOS, MOSE, and YouTube-VOS, see:

- [README_dataset.md](README_dataset.md)

---


## SAM3-LiteText Model Zoo & Weight Release


| Model | Text Encoder | Ctx | Text Params | Weights |
|-------|--------------|-----|-------------|---------|
| **SAM3-LiteText-S0-16** | MobileCLIP-S0 | 16 | 42.54M | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/sam3_litetext/efficient_sam3_image_encoder_mobileclip_s0_ctx16.pt)/ [GDrive](https://drive.google.com/file/d/1Eo81WYzfozFSvgvwlScGorUAIfMVPAFm/view?usp=sharing) |
| **SAM3-LiteText-S1-16** | MobileCLIP-S1 | 16 | 63.53M | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/sam3_litetext/efficient_sam3_image_encoder_mobileclip_s1_ctx16.pt)/ [GDrive](https://drive.google.com/file/d/1zL6x91PzvupHtZdA68jYip6yAUel8MMV/view?usp=sharing) |
| **SAM3-LiteText-L-16** | MobileCLIP2-L | 16 | 123.80M | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/sam3_litetext/efficient_sam3_image_encoder_mobileclip2_l_ctx16.pt)/ [GDrive](https://drive.google.com/file/d/1Mc4pk0FNCWwPTGoj1CCdAhkkNz02CUyY/view?usp=sharing) |

> All models use the **SAM3 ViT-H image encoder** (353.72M vision params). The text encoder parameters shown represent the distilled student replacing the original 353.72M text encoder, achieving up to **88% parameter reduction**.

---

##  Evaluation

##### Detailed Performance by Subset (Metric: CG_F1)

| Model | Ctx | MetaClip | SA1B | Crowd | Food | SptEq | Attr | Wiki | **Avg F1** | **MCC** | **pmF1** |
|-------|-----|----------|------|-------|------|-------|------|------|------------|---------|----------|
| **gDino-T** | - | 2.9 | 3.1 | 0.28 | 0.96 | 1.1 | 13.8 | 0.70 | 3.3 | 0.15 | 16.2 |
| **OWLv2** | - | 12.2 | 9.8 | 8.9 | 24.4 | 24.4 | 25.9 | 15.4 | 17.3 | 0.46 | 36.8 |
| **LLMDet-L** | - | 4.5 | 5.3 | 2.4 | 5.5 | 4.4 | 22.2 | 1.2 | 6.5 | 0.21 | 27.3 |
| **APE-D** | - | 12.6 | 2.2 | 7.2 | 22.7 | 31.8 | 26.7 | 11.6 | 16.4 | 0.40 | 36.9 |
| **DINO-X** | - | 17.2 | 19.7 | 12.9 | 30.1 | 28.4 | 31.0 | 9.7 | 21.3 | 0.38 | 55.2 |
| **Gemini 2.5**| - | 9.9 | 13.1 | 8.2 | 19.6 | 15.1 | 18.8 | 6.5 | 13.0 | 0.29 | 46.1 |
| **SAM3** | 32 | 47.3 | 53.7 | 61.1 | 53.4 | 65.5 | 54.9 | 42.5 | 54.1 | 0.82 | 66.1 |
| **SAM3-LiteText-S0** | 16 | 47.06 | 53.42 | 60.58 | 52.18 | 65.05 | 54.86 | 42.12 | 53.61 | 0.81 | 65.54 |
| **SAM3-LiteText-S1** | 16 | 47.18 | 53.58 | 60.76 | 52.43 | 65.28 | 55.02 | 42.35 | 53.80 | 0.81 | 65.72 |
| **SAM3-LiteText-L** | 16 | 47.24 | 53.66 | 60.88 | 52.65 | 65.49 | 55.19 | 42.54 | 53.95 | 0.81 | 65.87 |

> **Note:** This table shows performance of the released models, which were trained with a more extensive dataset mixture compared to the models reported in the paper. As a result, performance may differ slightly from the values in the associated publication.


</details>

---



## Citation

If you use SAM3-LiteText in your research, please cite:

```bibtex
@misc{zeng2026sam3litetextanatomicalstudysam3,
      title={SAM3-LiteText: An Anatomical Study of the SAM3 Text Encoder for Efficient Vision-Language Segmentation}, 
      author={Chengxi Zeng and Yuxuan Jiang and Ge Gao and Shuai Wang and Duolikun Danier and Bin Zhu and Stevan Rudinac and David Bull and Fan Zhang},
      year={2026},
      eprint={2602.12173},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.12173}, 
}
```

## License

This repository is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).

This project builds upon [SAM](https://github.com/facebookresearch/segment-anything), [SAM2](https://github.com/facebookresearch/sam2), [SAM3](https://github.com/facebookresearch/sam3), [EdgeSAM](https://github.com/chongzhou96/EdgeSAM), [EdgeTAM](https://github.com/facebookresearch/EdgeTAM), [EfficientTAM](https://github.com/yformer/EfficientTAM), [RepViT](https://github.com/THU-MIG/RepViT), [TinyViT](https://github.com/wkcn/TinyViT), [EfficientViT](https://github.com/mit-han-lab/efficientvit), and [MobileCLIP](https://github.com/apple/ml-mobileclip). Please refer to their respective licenses for usage terms.

## Acknowledgments

We gratefully acknowledge the [University of Bristol Isambard-AI supercomputer cluster](https://www.bristol.ac.uk/research/centres/bristol-supercomputing/articles/2025/isambard-ai-is-11th-fastest-supercomputer-in-the-world.html) for providing computational resources to this project. Special thanks to [Dr. Fan Aaron Zhang](https://fan-aaron-zhang.github.io/) for allocating resources and supporting this research.

---

## Users

Organizations and projects using EfficientSAM3:

<table>
  <tr>
    <td align="center" width="20%">
      <img src="https://github.com/SimonZeng7108/simonzeng7108.github.io/blob/main/efficientsam3/static/images/esa.png" alt="European Space Agency" height="80"><br>
      <a href="https://www.esa.int/">European Space Agency</a>
    </td>
  </tr>
</table>

> **Note:** If you're using EfficientSAM3 in your work, please acknowledge us in your publications or projects. We're happy to promote your work here! Contact us to be featured in this section.
