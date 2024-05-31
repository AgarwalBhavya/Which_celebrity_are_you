This project delves into the fascinating realm of facial recognition and machine learning to create a fun and interactive application. By leveraging pre-trained deep learning models and image processing techniques, users can discover which celebrity they resemble most based on their facial features!

Key Features:
Celebrity Image Recognition: Employs a pre-trained convolutional neural network (CNN) to extract features from celebrity images, creating a robust reference database.
User Image Processing: Guides users through uploading their photo, which is then preprocessed for compatibility with the CNN model.
Similarity Calculation: Computes the cosine similarity between the user's facial features and those of celebrities in the database, identifying the closest match.
Interactive Results: Presents the celebrity with the highest similarity score, potentially sparking amusement or surprise!

Technology Stack:
TensorFlow/Keras: Provides the foundation for deep learning model implementation and execution.
Scikit-learn: Offers functionalities for calculating cosine similarity or other distance metrics, if applicable.
NumPy: Handles numerical computations and array manipulations.
Pickle: Employed for efficiently storing and loading pre-processed data (e.g., celebrity features).

Usage:
Clone the repository: Run git clone https://github.com/AgarwalBhavya/Which-Celebrity-Are-You.git.
Activate virtual environment.
Run the script: Execute python test.py (or the appropriate script name).
Follow on-screen instructions: The script might prompt you to upload an image or provide the path to your pre-processed data files.
