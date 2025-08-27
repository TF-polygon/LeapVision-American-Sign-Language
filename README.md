# ASL Recognition with Patch-Based Network

This project implements an **American Sign Language (ASL) alphabet recognizer (26 letters)** using a **Patch-Based Network (PBN)** model.  
It provides training, evaluation, and real-time inference features, built with PyTorch and MediaPipe.

---

## **Project Structure**

```
├── asl_training.py # Model training script
├── evaluate_py.py # Model evaluation and visualization
├── inference_py.py # Real-time webcam inference
├── requirements.txt # Required dependencies
└── data/ASL-A/ # ASL dataset (train/val/test directory structure)
```
Example dataset directory structure:
```
data/ASL-A/
├── train/
│   ├── A/
│   ├── B/
│   └── ...
├── val/
│   ├── A/
│   ├── B/
│   └── ...
└── test/
    ├── A/
    ├── B/
    └── ...
```

Each letter folder should contain corresponding images (`.jpg`, `.png`, etc.).

---

## **Setup**

### Install Dependencies
```bash
pip install -r requirements.txt
pip install mediapipe
```
#### Requirements
```
torch>=2.0.0 
torchvision>=0.15.0
numpy>=1.21.0
pillow>=8.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
opencv-python>=4.5.0
tqdm>=4.62.0
```

## Train
```
python asl_training.py
```
By default, the script expects the dataset to be located at `data/ASL-A`, but you can change this path inside the script if needed. During training, images are resized to `224×224` pixels and undergo several data augmentation techniques such as random rotation, horizontal flipping, and color jittering to improve generalization. The model architecture is a simplified version of a patch-based Transformer encoder, where each image is split into patches and processed through a series of Transformer layers before classification into one of 26 ASL letters.

The script trains the network for 50 epochs using the SGD optimizer and a StepLR learning rate scheduler. After training, it saves the model weights to `pbn_asl_model.pth` and logs the training loss and validation accuracy history to `training_history.json`. It also prints metrics such as accuracy, precision, recall, and F1-score at each epoch so you can monitor progress.

## Evaluation
```
python evaluate_py.py
```
- Evaluates the model using the test set (data/ASL-A/test) <br>
- Outputs:
  - evaluation_results.json: accuracy, precision, recall, F1-score, per-class accuracy, etc.
  - confusion_matrix.png: confusion matrix visualization
  - per_class_accuracy.png: per-class accuracy bar chart
  - training_history.png: training history plot (if available)

## Inference
```
python inference_py.py
```
The script loads the saved `pbn_asl_model.pth` weights and opens your webcam feed. It uses MediaPipe to detect the location of your hand in each frame, crops the region of interest, and passes it through the model to predict the ASL letter. The predicted letter and its confidence score are drawn on the video feed. The window updates continuously and can be closed by pressing the `q` key. Accuracy in real-time inference may vary depending on lighting and background conditions, so ensure good lighting and clear visibility of your hand for the best results.

## Model Overview
- Patch-Based Network (PBN): Processes input images as patches via a Transformer Encoder.
- Input image size: 224x224
- Number of classes: 26 (A–Z)
- Hand ROI extraction: MediaPipe Hands
- Optimizer: SGD with StepLR scheduler

## Sample Results
- Best validation accuracy can reach around ~90% (depends on data and hyperparameters). 
- Confusion matrix and per-class accuracy plots are generated automatically.

## Notes
- Uses GPU automatically if available (cuda), otherwise runs on CPU.
- Data augmentation: random rotation, color jitter, horizontal flip, etc.
- Real-time inference accuracy may vary depending on lighting and background conditions.
