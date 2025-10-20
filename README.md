🍅 Tomato Disease Classification

This project demonstrates a deep learning model for classifying tomato plant diseases based on leaf images. The goal is to assist farmers and researchers in early disease detection to improve crop health and yield.

📂 Dataset

The dataset consists of images of healthy and diseased tomato leaves, categorized into 10 classes:

Tomato___Bacterial_spot

Tomato___Early_blight

Tomato___Late_blight

Tomato___Leaf_Mold

Tomato___Septoria_leaf_spot

Tomato___Spider_mites Two-spotted_mite

Tomato___Target_Spot

Tomato___Tomato_Yellow_Leaf_Curl_Virus

Tomato___Tomato_mosaic_virus

Tomato___Healthy

The dataset is divided into training and validation sets to evaluate model performance effectively.

🧠 Model Architecture

The model uses transfer learning with a pre-trained DenseNet121 convolutional base, followed by custom dense layers for classification.

Architecture Summary:

Base Model: DenseNet121 (pre-trained on ImageNet)

Top Layers:

Batch Normalization

Dense (256 units, ReLU activation)

Dropout (rate = 0.35)

Batch Normalization

Dense (120 units, ReLU activation)

Dense (10 units, Softmax activation) — for 10-class classification

⚙️ Training Details

Optimizer: Adam

Learning Rate: 0.0001

Loss Function: Categorical Crossentropy

Metric: Accuracy

Early Stopping: Enabled (patience = 0)

Epochs: 20

📈 Results

After training for 20 epochs, the model achieved:

Validation Accuracy: 94.8%

Training and validation accuracy/loss plots are provided to visualize the model’s performance over time.

📊 Future Improvements

Implement data augmentation to improve model generalization.

Deploy the model as a web or mobile application for real-time disease detection.

Experiment with other architectures like EfficientNet or ResNet50.

🏁 Conclusion

This project successfully demonstrates a tomato leaf disease classification system with high accuracy using transfer learning. It can serve as a foundation for building practical tools that help farmers identify plant diseases early.
