# Cat vs Dog Image Classifier

## Project Overview

This project implements a convolutional neural network (CNN) to classify images of cats and dogs. The model is built using TensorFlow and Keras, leveraging transfer learning with the MobileNetV2 architecture.

## Dataset

The dataset consists of images of cats and dogs, split into training and validation sets. The images are preprocessed and augmented to improve model generalization.

## Approach

1. **Data Preprocessing**:

   - Images are resized to 160x160 pixels
   - Pixel values are normalized to the range [-1, 1]
   - Data augmentation techniques are applied, including random flips and rotations

2. **Model Architecture**:

   - Transfer Learning: MobileNetV2 is used as the base model
   - Global Average Pooling is applied to the output of the base model
   - A dense layer with 1024 units and ReLU activation is added
   - The final output layer uses sigmoid activation for binary classification

3. **Training**:

   - Binary crossentropy loss function
   - Adam optimizer with a learning rate of 0.0001
   - Early stopping to prevent overfitting
   - Model checkpointing to save the best model

4. **Evaluation**:
   - The model is evaluated on a separate validation set
   - Accuracy and loss metrics are used to assess performance

## Results

The model achieves high accuracy on both training and validation sets, demonstrating effective learning of cat and dog features.

## Future Improvements

- Fine-tuning the MobileNetV2 base model
- Experimenting with other pre-trained models (e.g., ResNet, Inception)
- Implementing cross-validation for more robust evaluation
- Exploring advanced data augmentation techniques

## Dependencies

- TensorFlow
- Keras
- NumPy
- Matplotlib

## Usage

1. Clone the repository
2. Install the required dependencies
3. Run the Jupyter notebook `fcc_cat_dog.ipynb`

## License

This project is open-source and available under the MIT License.
