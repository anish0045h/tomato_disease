#Tomato Disease Classification
This project demonstrates a deep learning model for classifying tomato plant diseases based on images.

Dataset
The dataset used in this project contains images of healthy and diseased tomato leaves, categorized into 10 classes:

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
The dataset is split into training and validation sets.

Model Architecture
The model utilizes transfer learning with a pre-trained DenseNet121 convolutional base. The convolutional base is followed by:

Batch Normalization
A Dense layer with 256 units and ReLU activation
Dropout with a rate of 0.35
Batch Normalization
A Dense layer with 120 units and ReLU activation
A final Dense layer with 10 units and softmax activation for classification
Training
The model was trained using the Adam optimizer with a learning rate of 0.0001. The loss function used was categorical crossentropy, and the model was evaluated based on accuracy. Early stopping with patience 0 was used to prevent overfitting.

Results
After training for 20 epochs, the model achieved a validation accuracy of 94.8%. The training and validation loss and accuracy over the epochs are visualized in the plots below.

How to use the model
Load the model: Load the saved model file (tomato_disease_model.keras).
Preprocess new images: Resize new images to 256x256 pixels, convert them to a numpy array, expand dimensions to include the batch size, and normalize the pixel values by dividing by 255.0.
Make predictions: Use the loaded model to predict the class of the preprocessed image.
