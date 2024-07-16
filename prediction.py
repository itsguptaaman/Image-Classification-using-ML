import os
import traceback
import pickle
import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(ROOT_PATH, 'models')
model_path = os.path.join(model_dir, 'models.pkl')

def load_model(model_path):
    try:
        with open(model_path, 'rb') as file:
            model_data = pickle.load(file)
        return model_data
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()

def process_image(img, target_size=(128, 128)):
    try:
        img = cv2.resize(img, target_size)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eq_img = cv2.equalizeHist(gray_img)
        return eq_img
    except Exception as err:
        print(f"Error processing image: {err}")
        return None

def extract_features(img):
    histogram = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5).flatten()
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5).flatten()
    hog_features, _ = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
    lbp_histogram = np.histogram(lbp, bins=np.arange(0, 27), range=(0, 26))[0]
    combined_features = np.hstack((histogram, sobel_x, sobel_y, hog_features, lbp_histogram))
    return combined_features

# Prediction Function
def predict_image(image_path, clf, pca, scaler, labels):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Unable to read the image.")
        return None

    processed_img = process_image(img)
    if processed_img is None:
        print("Error: Unable to process the image.")
        return None

    features = extract_features(processed_img).reshape(1, -1)
    features_scaled = scaler.transform(features)
    features_pca = pca.transform(features_scaled)
    pred_encoded = clf.predict(features_pca)
    pred_label = labels[pred_encoded[0]]
    return pred_label

if __name__ == "__main__":
    model_data = load_model(model_path)
    clf = model_data['classifier']
    pca = model_data['pca']
    scaler = model_data['scaler']
    labels = model_data['label_encoder']
    
    example_image_path = os.path.join(ROOT_PATH, 'test_images', 'test6.jpg')  # Adjust the path as needed
    predicted_label = predict_image(example_image_path, clf, pca, scaler, labels)
    print(f"Predicted label: {predicted_label}")
