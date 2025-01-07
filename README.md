# LLaVA-Based Wildfire Detection

This repository contains Dockerfiles and Python scripts to build and run [LLaVA](https://github.com/haotian-liu/LLaVA) models for image-based wildfire detection. It uses variants of \(`vip-llava-13b-hf`, `llava-1.5-7b-hf`\) that can be run in 4-bit or 8-bit precision on NVIDIA GPUs, enabling faster inference while conserving GPU memory.

## Table of Contents

- [Overview](#overview)  
- [Contents of This Repository](#contents-of-this-repository)  
  - [Dockerfiles](#dockerfiles)  
  - [Requirements Files](#requirements-files)  
  - [Python Scripts](#python-scripts)  
- [How to Build and Run](#how-to-build-and-run)  
  - [Building the Docker Image](#building-the-docker-image)  
  - [Running the Docker Container](#running-the-docker-container)  
  - [Polaris Version](#polaris-version)  
- [File-Specific Usage Notes](#file-specific-usage-notes)  
- [Directory Structure](#directory-structure)  
- [Contributing](#contributing)  
- [License](#license)  
- [Acknowledgments](#acknowledgments)

---

## Overview

The goal of this project is to detect indications of wildfire (e.g. smoke or flames) from images using LLaVA-based vision-language models. The scripts can:
- Classify images as likely containing wildfire smoke/fire or not.
- Split the input image (e.g., into an optical camera image and an IR pseudo-color image) and process each half separately.
- Generate outputs and internal hidden representations for advanced analysis.
- Save checkpoints and outputs to persistent storage for further review.

## Contents of This Repository

### Dockerfiles

1. **Dockerfile**  
   - Based on `nvcr.io/nvidia/pytorch:24.01-py3`.
   - Installs the required libraries from `requirements.txt`.
   - Sets environment variables for the Hugging Face Transformers cache.
   - Copies all repository files into the container.
   - (Commented out lines show examples of alternative base images and usage hints.)

2. **Dockerfile_polaris**  
   - Similar to `Dockerfile` but uses `nvcr.io/nvidia/pytorch:22.06-py3` as the base image.
   - Installs libraries specified in `requirements_polaris.txt`.

### Requirements Files

1. **requirements.txt**  
   Contains the standard Python dependencies:
   ```text
   numpy
   transformers[torch]
   accelerate
   bitsandbytes
   ```

2. **requirements_polaris.txt**  
   Contains additional dependencies for the Polaris environment:
   ```text
   numpy
   transformers[torch]
   accelerate
   bitsandbytes
   sentencepiece
   mpi4py
   ```

### Python Scripts

Below is a quick summary of each script. All scripts assume the `/images` folder contains the images to process, and `/RESULTS` is where outputs are stored. Many of these scripts share a similar structure: they load a model, loop over images, run inference, and save results/checkpoints.

1. **`konza_run_vip-llava-13b-hf_model.py`**  
   - Uses the VIP LLaVA 13B model \(`llava-hf/vip-llava-13b-hf`\) to analyze images for wildfire indicators.  
   - Splits the image into two halves (RGB + IR), processes them separately, and logs results to `/RESULTS`.

2. **`run_llava-1.5-7b-hf_model.py`**  
   - Uses `llava-hf/llava-1.5-7b-hf` for classification.  
   - Scans the `/images` directory, classifies each image with a single prompt, and writes results in CSV format to `/RESULTS`.  
   - Copies images predicted to contain fire into `/RESULTS/FIRE_IMAGES/`.

3. **`run_vip-llava-13b-hf_model.py`**  
   - Similar to `konza_run_vip-llava-13b-hf_model.py`, but with different prompts and logic flow.  
   - If it detects smoke (or potential wildfire), it copies the file to `/RESULTS/FIRE_IMAGES/` and logs extended details for those images.

4. **`run_internal_llava-1.5-7b-hf_model.py`**  
   - Also uses `llava-hf/llava-1.5-7b-hf`.  
   - In addition to generating a textual response, it captures and saves internal hidden-state representations from the model.  
   - Writes them out as tensors (`.pt` files) for later analysis.

5. **`run_internal_vip-llava-13b-hf_model.py`**  
   - Similar to `run_internal_llava-1.5-7b-hf_model.py`, but based on the VIP LLaVA 13B model.  
   - Captures internal feature representations for the original, RGB-cropped, and IR-cropped images.

Each script includes functions to:
- Load existing checkpoints and partial outputs (so you can resume from the last processed image).  
- Save updated checkpoints, textual results, and internal representations.  
- Perform inference on the entire image or on specific sub-regions (RGB vs. IR).

---

## How to Build and Run

### Building the Docker Image

To build the Docker image from the **Dockerfile** (using the `nvcr.io/nvidia/pytorch:24.01-py3` base), run:

```bash
# From the repository root:
docker build -t hfsandbox:latest -f Dockerfile .
```

If you want to use the **Polaris**-compatible image (using the `nvcr.io/nvidia/pytorch:22.06-py3` base):

```bash
docker build -t hfsandbox:polaris -f Dockerfile_polaris .
```

### Running the Docker Container

You can run the container with GPU access and mount local directories for images, HF cache, and results:

```bash
sudo docker run --gpus all -it --rm \
    -v /path/to/images:/images \
    -v /path/to/huggingface/cache:/hf_cache \
    -v /path/to/results:/RESULTS \
    hfsandbox:latest
```

Within the container, you can run any of the scripts (e.g., `konza_run_vip-llava-13b-hf_model.py`) to process `/images` and store outputs in `/RESULTS`.

### Polaris Version

If you plan to run this on the [Polaris supercomputer](https://www.alcf.anl.gov/polaris) (or another HPC environment) where you need `mpi4py` and a slightly older CUDA/PyTorch stack, use the container built from `Dockerfile_polaris` and the `requirements_polaris.txt` file.

---

## File-Specific Usage Notes

- **`requirements_polaris.txt`** includes `mpi4py` and `sentencepiece`, which might be unnecessary for local runs but required on HPC systems like Polaris.  
- In each script, you can adjust the prompts or classification logic (e.g., changing from “Is there fire?” to “Is there smoke?”) according to your application needs.  
- By default, the scripts look for `checkpoint.txt` in `/RESULTS/` to detect already processed files. If you remove or rename it, the script will reprocess everything from scratch.

---

## Directory Structure

A typical layout after cloning this repository and building the container could look like:

```
.
├── Dockerfile
├── Dockerfile_polaris
├── requirements.txt
├── requirements_polaris.txt
├── konza_run_vip-llava-13b-hf_model.py
├── run_internal_llava-1.5-7b-hf_model.py
├── run_internal_vip-llava-13b-hf_model.py
├── run_llava-1.5-7b-hf_model.py
├── run_vip-llava-13b-hf_model.py
├── README.md
└── ...
```

- `/images` (mounted at runtime)  
  - The directory containing images (`.jpg`) for inference.  
- `/RESULTS` (mounted at runtime)  
  - Where the scripts will store results (`output.csv`), checkpoints (`checkpoint.txt`), subfolders like `FIRE_IMAGES/`, and `.pt` files with hidden states if applicable.  
- `/hf_cache` (mounted at runtime)  
  - Caches model weights from Hugging Face so you don’t redownload them each time.

---

## Contributing

If you want to contribute:
1. Fork this repository.
2. Create a new feature branch.
3. Make your changes and test them.
4. Submit a pull request describing your changes.

Feel free to open an issue if you find bugs or want to request a feature!

---

## License

This project does not have a specified license. Please note that the underlying LLaVA and VIP-LLaVA models have their own licenses. Refer to their respective repositories for more details.

---

## Acknowledgments

- [LLaVA](https://github.com/haotian-liu/LLaVA) team for open-sourcing the vision-language alignment models.  
- [Hugging Face](https://huggingface.co/) for hosting model checkpoints.  
- [NVIDIA NGC](https://ngc.nvidia.com/) for providing base Docker images with PyTorch support.

