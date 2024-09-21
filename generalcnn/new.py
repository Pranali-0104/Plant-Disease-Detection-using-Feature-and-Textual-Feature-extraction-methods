import streamlit as st
import tensorflow as tf
import numpy as np

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.
    """)

elif app_mode == "About":
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this github repo.
                This dataset consists of about 87K RGB images of healthy and diseased crop leaves which is categorized into 38 different classes.
                """)

elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    
    if st.button("Show Image"):
        st.image(test_image, width=4, use_column_width=True)
    
    # Predict button
    if st.button("Predict"):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        
        # Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                    'Tomato___healthy']
        
        # Get the predicted class label
        predicted_class = class_name[result_index]
        st.write(f"Predicted Class: {predicted_class}")
        
        # Show detailed info based on the predicted class
        if predicted_class == 'Apple___Cedar_apple_rust':
            st.success("""Symptoms: Yellow-orange spots on leaves.
            Early leaf drop in severe cases. Affects both apple and cedar trees, causing damage to leaves, fruit, and twigs.""")
            st.success("""Chemical Treatment: Fungicides: Apply fungicides containing myclobutanil or propiconazole during early spring.
            Application: Start applications at bud break and repeat every 7-10 days.""")
            st.success("""Organic Treatment: Neem Oil: Neem oil sprays can help manage fungal infections.
            Garlic Spray: Apply garlic-based sprays to inhibit fungal growth.""")
            st.success("""Precautions: Remove and destroy infected leaves and plant debris. Avoid planting susceptible trees near junipers.""")
        
        elif predicted_class == 'Apple___Apple_scab':
            st.success("""Symptoms: Dark, olive-green spots on leaves and fruit.
            Premature leaf drop and fruit deformities.""")
            st.success("""Chemical Treatment: Apply fungicides like captan or sulfur during early spring.""")
            st.success("""Organic Treatment: Compost Tea: Spray compost tea to strengthen plant immunity.""")
            st.success("""Precautions: Remove infected leaves and improve air circulation around the tree.""")
                
        elif predicted_class == 'Corn_(maize)___Common_rust':
            st.success("""Symptoms: Reddish-brown pustules on leaves, often forming in clusters.
            Severely affected leaves may wither and die prematurely.""")
            st.success("""Chemical Treatment: Fungicides: Use products containing azoxystrobin or pyraclostrobin.""")
            st.success("""Organic Treatment: Neem Oil: Regular neem oil sprays can reduce rust development.""")
            st.success("""Precautions: Practice crop rotation and ensure proper plant spacing to minimize humidity.""")
            
        elif predicted_class == 'Potato___Early_blight':
            st.success("""Symptoms: Dark brown spots with concentric rings on leaves, starting from the bottom.
            Affected leaves may drop, leading to reduced yield.""")
            st.success("""Chemical Treatment: Fungicides: Apply products containing mancozeb or chlorothalonil.""")
            st.success("""Organic Treatment: Copper Spray: Use copper-based sprays to control early blight.""")
            st.success("""Precautions: Rotate crops and avoid overhead irrigation to reduce moisture on the leaves.""")
        
        elif predicted_class == 'Tomato___Early_blight':
            st.success("""Symptoms: Dark lesions with concentric rings on leaves, often starting from older leaves.
            Leads to early defoliation and poor fruit development.""")
            st.success("""Chemical Treatment: Use fungicides like chlorothalonil or copper-based products.""")
            st.success("""Organic Treatment: Compost Tea: Spray compost tea to strengthen plant immunity and reduce disease severity.""")
            st.success("""Precautions: Rotate crops and maintain proper plant spacing to ensure good air circulation.""")
            
        elif predicted_class == 'Potato___healthy':
            st.success("""Regular Monitoring: Ensure plants receive balanced nutrients and proper irrigation to stay healthy.
            Good Agricultural Practices: Use crop rotation and avoid waterlogging to prevent diseases.""")
        
        elif predicted_class == 'Tomato___healthy':
            st.success("""Regular Monitoring: Monitor plants for pests and diseases and use preventative measures.
            Good Agricultural Practices: Maintain proper watering, fertilization, and pruning techniques.""")
        elif predicted_class == 'Tomato___Tomato_Yellow_Leaf_Curl_Virus':
            st.success("""Symptoms: Leaves become curled, wrinkled, and yellow. Plants exhibit stunted growth and reduced yield.""")
            st.success("""Chemical Treatment: No effective chemical treatment is available for this viral disease.""")
            st.success("""Organic Treatment: Resistant Varieties: Plant virus-resistant tomato varieties.""")
            st.success("""Precautions: Remove and destroy infected plants to prevent the spread of the virus.""")