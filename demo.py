import streamlit as st
import time
from PIL import Image
from pickle import load
from skimage import feature
from bidict import bidict
import gdown 
import os
st.header("BHXLA - HCMUS - 2024-2025")

def hog_features(img):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    hog, hog_image = feature.hog(
        img, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), block_norm="L2-Hys",
        visualize=True, transform_sqrt=True, channel_axis=-1
    )
    return hog

labels_dict = bidict({"Accessibility":0, "Female":1, "Male":2, "No Smoking":3, "Wifi":4})
file = st.file_uploader('Tải ảnh lên', type=['jpg', 'jpeg', 'png', 'jfif'])

# URL chia sẻ Google Drive 
url = 'https://drive.google.com/uc?id=<1-cffp5jtQq9y6QBpJmY72fSTy8B07bK_>' 
output = 'svm_model_hog.pkl' 
# Tải tệp mô hình từ Google Drive nếu chưa tồn tại 
if not os.path.exists(output): 
    gdown.download(url, output, quiet=False)
    
if file is not None: 
    # Đọc ảnh trực tiếp từ bộ nhớ 
    img = Image.open(io.BytesIO(file.getvalue())) 
    feature_vector = hog_features(img)

    column_1, column_2 = st.columns(2)
    with column_1:
        st.header("Ảnh đầu vào")
    with column_2:
        st.header("HOG + SVM")

    with column_1:
        img = Image.open(path_to_image)
        st.image(img)
    with column_2:
        with open("D:/IndoorSignages-20241113T120343Z-001/IndoorSignages/no_augmentation/models/svm_model_hog.pkl", "rb") as file:
            model = load(file)
        y_pred = model.predict(feature.reshape(1, -1))
        with st.spinner("Đang dự đoán"):
            time.sleep(1)
        st.header(labels_dict.inverse[y_pred[0]])
