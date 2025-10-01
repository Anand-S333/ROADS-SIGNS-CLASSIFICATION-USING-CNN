Project Documentation: CNN-Based Road Signs Classification 
Dataset details: 
• Images: The road sign images that will be classified. 
• Label: 30 classes or categories of road signs (e.g., speed limit, stop sign). 
• Dataset Split: Distribution of data for training and testing. 
Step-by-Step Instructions: 
Step 1: Data Collection and Preprocessing 
1. Collecting the Dataset 
• Download road signs dataset. 
• Labels are given in the excel sheet. 
2. Data Preprocessing 
• Resizing Images: Resize the images to a consistent shape  
• Normalization: Normalize pixel values to be between 0 and 1 by dividing by 
255. 
• Label Encoding: Encode labels as integers or one-hot vectors for 
classification. 
3. Data Augmentation  
o Apply data augmentation techniques such as rotation, scaling, flipping, and 
cropping to increase the dataset's diversity and robustness. 
Step 2: Build the Convolutional Neural Network (CNN) 
1. CNN Architecture Design 
• Input Layer: Define the input shape based on the image dimensions (e.g., 
64x64x3 for RGB images). 
• Convolutional Layers: Add several convolutional layers to extract features 
from the images. 
• Pooling Layers: Implement max-pooling layers to downsample the feature 
maps.
Fully Connected Layers: Add dense layers at the end to classify the image 
based on extracted features. 
• Activation Functions: Use ReLU for hidden layers and softmax for the output 
layer (for multi-class classification). 
2. Compilation of the Model 
• Choose an optimizer (e.g., Adam, SGD), loss function (e.g., categorical cross
entropy), and evaluation metrics (e.g., accuracy). 
Step 3: Training the Model 
1. Splitting the Data 
• Split the data into training, validation, and test sets (e.g., 70% training, 15% 
validation, 15% testing). 
2. Model Training 
• Trained the CNN on the training dataset while monitoring performance on the 
validation set. 
• Used techniques like early stopping and checkpoints to avoid overfitting and 
ensured the model’s generalization. 
Step 4: Model Evaluation and Testing 
1. Evaluating Model Performance 
• Evaluate the trained model on the test set using metrics like accuracy, 
precision, recall, and F1-score. 
• Visualize confusion matrices to show the performance across different road 
sign classes. 
2. Testing the Model with New Images 
• Test the model on new, unseen images to verify its real-world performance. 
• Output predictions and compare them with actual labels. 
Step 5: Hyperparameter Tuning and Model Optimization 
1. Grid Search / Random Search 
• Implemented hyperparameter tuning to optimize learning rate, number of filters, 
kernel size, etc. 
2. Model Optimization 
• Experiment with different architectures (e.g., adding more layers, changing the 
number of neurons in fully connected layers). 
• Applied regularization techniques (e.g., dropout) to improve model 
performance. 
Step 6: Create an UI for the trained model using Streamlit 
• Provided information on accuracy and class predictions for new test data. 
Final Deliverables 
1. Trained Model 
• The trained CNN model saved in a compatible format for future use. 
2. Performance Metrics 
• Provided detailed performance metrics, including accuracy, precision, recall, 
F1-score, and confusion matrix. 
3. Visualizations 
• Included graphs showing training and validation accuracy/loss, confusion 
matrices, and model performance per class.
