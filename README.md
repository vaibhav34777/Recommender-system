## Content-Based Recommender System Using Deep Learning
This project implements a content-based recommendation system using deep learning techniques in TensorFlow/Keras. The system recommends items to users based on their features by learning vector representations of users and items and computing their similarity.

### Features
##### Deep Neural Networks: Separate neural networks for users and items to learn latent representations.
##### Content-Based Filtering: Uses features of users and items to compute recommendations.
##### L2 Normalization: Ensures the similarity is computed effectively in a unit hyperspace.
##### Dot Product Similarity: Measures the closeness between user and item embeddings.
##### Custom Training: Supports custom loss functions and optimizers for fine-tuning the recommendation performance.
### Dataset
The data set is derived from the [MovieLens ml-latest-small](https://grouplens.org/datasets/movielens/latest/) dataset. 

The system requires a dataset containing:

User Features: Numerical or categorical data representing user profiles (e.g., age, location, preferences).
Item Features: Attributes of items (e.g., genre, price, category).
Interaction Data: User-item interactions for training the model (e.g., ratings, clicks).
### Model Architecture
The recommendation system is built with two separate neural networks:

#### User Neural Network:
Layers: Dense layers with ReLU activations to learn user embeddings.
#### Item Neural Network:
Layers: Dense layers with ReLU activations to learn item embeddings.
#### Final Layer
Dot Product Similarity: Computes the dot product of user and item embeddings as the recommendation score.
#### Loss Function
Custom loss functions like MSE (Mean Squared Error) or BCE (Binary Crossentropy) can be used based on the task.

### Installation
Prerequisites
Python 3.7 or higher,
TensorFlow 2.x,
Numpy,
Pandas,
Scikit-learn (optional for preprocessing)
### Acknowledgments
TensorFlow and Keras for deep learning.
Scikit-learn for preprocessing utilities.
Inspiration from collaborative filtering and deep learning methods in recommendation systems.
