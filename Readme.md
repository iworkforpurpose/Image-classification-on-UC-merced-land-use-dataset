# 🧠 CNN Architectures Comparison: LeNet, AlexNet, VGG19 & ResNet50

This project implements and compares four popular Convolutional Neural Network (CNN) models — LeNet-5, AlexNet, VGG19, and ResNet50 — on a classification task using PyTorch. The models are trained, evaluated, and visualized with custom tools and plots.

---

## 📁 Project Structure

├── main.py # Main training/testing pipeline
├── config.py # Configuration and hyperparameters 
├── models/ │ ├── lenet.py │ ├── alexnet.py │ ├── vgg.py │ └── resnet.py 
├── utils/ │ ├── data_utils.py # Data loaders │ 
             ├── visualization.py # Training curves, confusion matrix, metrics 
├── outputs/ # Model weights and results (excluded in .gitignore) 
├── requirements.txt # Python dependencies 
└── README.md # This file


---

## 🚀 Models Implemented

- ✅ LeNet-5  
- ✅ AlexNet  
- ✅ VGG19  
- ✅ ResNet50 (Pretrained & fine-tuned)

---

## 📊 Visualizations & Metrics

- 📉 Training & validation loss/accuracy curves  
- 📌 Normalized confusion matrices  
- 📊 Per-class precision, recall, and F1-score  
- ❌ Top misclassified classes  

All visualizations are implemented in `utils/visualization.py` using `matplotlib` and `seaborn`.

---

## 📦 Installation

git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt

🏁 How to Run
To train and test all models:
python main.py

You can also edit main.py to run a specific model individually.

🧪 Sample Results

===== Testing LeNet5 =====
Test Loss: 0.0014, Test Accuracy: 1.0000

===== Testing AlexNet =====
Test Loss: 0.0000, Test Accuracy: 1.0000

===== Testing VGG19 =====
Test Loss: 0.0000, Test Accuracy: 1.0000

===== Testing ResNet50 =====
Test Loss: 0.0002, Test Accuracy: 1.0000
⚠️ Note: These results may indicate overfitting or an easy dataset. Always validate on a diverse test set.

🧰 Tools & Libraries
PyTorch

torchvision

matplotlib

seaborn

scikit-learn

numpy

📌 To-Do
 Add TensorBoard/W&B logging

 Add support for custom datasets

 Save/load trained model checkpoints

 Try model ensembling

 🧑‍💻 Author
Vighnesh Nama
AI/ML Student
Driven by clean engineering 🧠💻
