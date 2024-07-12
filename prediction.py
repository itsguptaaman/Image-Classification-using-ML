
import os
import traceback
import pickle
import numpy as np
import cv2


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


def extract_color_intensity(image):
    # Calculate the mean color intensity for each channel (R, G, B)
    mean_intensity = image.mean(axis=(0, 1))
    return mean_intensity


# Prediction Function
def predict_image(image_path, clf, pca, le):
    img = cv2.imread(image_path)
    if img is not None:
        img = cv2.resize(img, (128, 128))  # Resize image to a fixed size
        img_flattened = img.reshape(1, -1)  # Flatten the image
        img_reduced = pca.transform(img_flattened)  # Apply PCA
        color_intensity = extract_color_intensity(img)  # Extract color intensity features
        combined_features = np.hstack((img_reduced, color_intensity.reshape(1, -1)))  # Combine features
        pred_encoded = clf.predict(combined_features)  # Predict
        pred_label = le.inverse_transform(pred_encoded)  # Decode label
        return pred_label[0]
    else:
        return None


if __name__ == "__main__":
    model_data = load_model(model_path)
    clf = model_data['classifier']
    pca = model_data['pca']
    le = model_data['label_encoder']
    example_image_path = os.path.join(ROOT_PATH, os.path.join(ROOT_PATH, 'test_images', 'test6.jpg'))
    predicted_label = predict_image(example_image_path, clf, pca, le)
    print(f"Predicted label: {predicted_label}")