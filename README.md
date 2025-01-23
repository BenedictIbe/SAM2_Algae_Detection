# SAM2 Segmentation Framework

This repository provides a framework for semantic segmentation using the SAM2 (Segment Anything Model v2). SAM2 extends the capabilities of the original SAM model, supporting fine-tuning and segmentation with higher accuracy for tasks like harmful algae bloom detection and water segmentation.


# Table of Contents
 - Overview
 - Features
 - Requirements
 - Installation
 - Usage
    - Training
    - Inference and Visualization
 - Evaluation
 - Results
 - Contributing
 - License

# Overview
The SAM2 Segmentation Framework leverages the SAM2 architecture for high-performance segmentation. It is particularly useful for environmental monitoring tasks, such as detecting harmful algae blooms (HABs) in aquatic environments.

# Key highlights
 - Fine-tunable SAM2 model for specific segmentation tasks.
 - Dice score evaluation for segmentation accuracy.
 - Visualization of predicted masks overlaid on the original image.

# Sample visualizations:
- Original Image
- Ground Truth Mask
- Predicted Mask

# Features
**SAM2 Model Integration:** Extendable segmentation architecture with pretrained weights.
**Bounding Box and Prompt-Based Segmentation:** Fine-tuning and inference using bounding boxes or prompts.
**Dice Score Evaluation:** Automated computation of Dice scores to evaluate segmentation quality.
**Visualization:** Overlay segmented areas with shading for intuitive visualization.
**Custom Dataset Support:** Easily adaptable to custom datasets with appropriate annotations.

# Requirements
The following libraries are required to run this project:

 - Python 3.8+
 - PyTorch 1.10+
 - OpenCV 4.x
 - Matplotlib
 - Pandas
 - SAM2 (cloned from the Facebook Research repository)
 - Installation

# Clone the repository:
    git clone https://github.com/yourusername/sam2-segmentation.git
    cd sam2-segmentation

# Install dependencies:
    pip install -r requirements.txt

Download pretrained SAM2 weights:
 - Place the SAM2 checkpoint (e.g., sam2_hiera_large.pt) in the checkpoints directory.

# Usage
**Training**
**Prepare your dataset:**
 - Ensure a CSV file (e.g., train.csv) with columns Original_Image_Path and Segmentation_Mask_Path.

# Example:**
    Original_Image_Path,Segmentation_Mask_Path
    data/images/image1.jpg,data/masks/mask1.png
    data/images/image2.jpg,data/masks/mask2.png

# Train the model:
    python train_sam2.py --config sam2_hiera_l.yaml --checkpoint checkpoints/sam2_hiera_large.pt --train_csv train.csv --num_iterations 3000 --save_interval 500

Training parameters:

 - --config: Path to the SAM2 configuration file.
 - --checkpoint: Path to the pretrained SAM2 weights.
 - --train_csv: Path to the training data CSV file.
 - --num_iterations: Number of training iterations.
 - --save_interval: Interval to save model checkpoints.

# Inference and Visualization
    python infer_sam2.py --test_csv test.csv --checkpoint checkpoints/sam2_final_model.pth --config sam2_hiera_l.yaml --max_visualizations 10

**Parameters:**
 - --test_csv: Path to the testing data CSV file.
 - --checkpoint: Path to the trained SAM2 model checkpoint.
 - --config: Path to the SAM2 configuration file.
 - --max_visualizations: Number of images to visualize.

# Sample Output:
 - Predicted segmentation masks are shaded and displayed over the input images.
 - Average Dice score is printed to the console for evaluation.

# Evaluation
The framework evaluates the segmentation performance using the Dice score, a common metric for measuring overlap between predicted and ground-truth masks. Results are reported per sample and as a mean over the test dataset.

# Results
 - Metric	Value
 - Mean Dice	0.92


# Contributing
We welcome contributions to improve this project. Feel free to open issues or submit pull requests.

To contribute:
 - Fork the repository.
# Create a new branch:
    git checkout -b feature-name

# Commit your changes:
    git commit -m "Add new feature"

# Push to your branch:
    git push origin feature-name

Submit a pull request.

# License
This project is licensed under Benedict Ibe.
