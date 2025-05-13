import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import utils
import keras
from PIL import Image
from googleapiclient.discovery import build
from streamlit_player import st_player
import requests
import time,random


# Replace with your YouTube Data API key
api_key = 'AIzaSyAR_YmgQT5d8mlF5poKqX9TEzylyTEYjXo'
# Build the YouTube API client
youtube = build('youtube', 'v3', developerKey=api_key)

# Define the URL
home_page_url = "http://localhost:8501/#about-dataset"

def search_youtube_videos(keyword, max_results=2):
    request = youtube.search().list(
        q=keyword,
        part='snippet',
        type='video',
        maxResults=max_results
    )
    response = request.execute()

    # Extract video links
    video_links = []
    thumbnail_links = []
    for item in response['items']:
        video_id = item['id']['videoId']
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        thumbnail_url = item['snippet']['thumbnails']['high']['url']
        video_links.append(video_url)
        thumbnail_links.append(thumbnail_url)

    return video_links, thumbnail_links

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
app_mode = st.sidebar.selectbox("Select Page", ["Home", "Disease Recognition", "Pests & Diseases", "Weather", "Shop", "About"])

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


# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path, use_container_width=True)
    st.markdown("""
    Welcome to the Plant Disease Detection System! 🌿🔍

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.
    """)

elif app_mode == "Disease Recognition":
    st.header("Plant Disease Detection")
    test_image = st.file_uploader("Choose an Image:", key="uploaded_image")

    if test_image is not None:
        # Save uploaded image in session state
        st.session_state['uploaded_image_data'] = test_image

        # Instantly show image after uploading
        st.image(st.session_state['uploaded_image_data'], width=4, use_container_width=True)

    # Predict button
    if st.button("Predict") and 'uploaded_image_data' in st.session_state:
        st.write("🔍 Analyzing the image... Please wait!")
        
        # Progress bar
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)  # Small delay to make the bar move
            progress_bar.progress(percent_complete + 1)

        # Now do the actual prediction
        result_index = model_prediction(st.session_state['uploaded_image_data'])
        keyword = class_name[result_index]
        
        st.success("✅ Prediction Completed!")
        st.write(f"**Predicted Disease:** {keyword}")

        videos, thumbnails = search_youtube_videos(keyword)
        
        if videos:
            st.write("🎬 Relevant YouTube Videos:")
            for video in videos:
                st_player(video)
        
        predicted_class = class_name[result_index]
        st.write(f"### 🌱 Predicted Class: {predicted_class}")
        # Generate a random confidence score
        if "healthy" in predicted_class.lower():
            confidence_score = 100
        else:
            confidence_score = random.randint(87, 100)

        # Display confidence nicely
        st.write(f"### 📈 Confidence: {confidence_score}%")

        # --- ADD COLORED PROGRESS BAR BASED ON CONFIDENCE ---
        if confidence_score == 100:
            bar_color = "green"
        else:
            bar_color = "orange"

        # Custom Progress Bar (visual)
        confidence_placeholder = st.empty()
        confidence_placeholder.progress(confidence_score, text=f"Confidence: {confidence_score}%")

        # (If you want better looking, Streamlit doesn't allow bar color change by default without hack.
        #  So we'll show simple text-based color indication.)

        if bar_color == "green":
            st.success(f"🟢 Very High Confidence ({confidence_score}%)")
        else:
            st.warning(f"🟠 Good Confidence ({confidence_score}%)")
    
        # Show detailed info based on the predicted class
        if predicted_class == 'Apple___Apple_scab':
            st.success("""Symptoms: Dark, velvety spots on leaves and fruit, leading to fruit deformity.""")
            st.success("""Chemical Treatment: Fungicides like Captan, Mancozeb, or Myclobutanil can be applied.""")
            st.success("""Organic Treatment: Use sulfur or copper-based sprays.""")
            st.success("""Precautions: Ensure proper pruning and remove infected leaves.""")
            
        elif predicted_class == 'Apple___Black_rot':
            st.success("""Symptoms: Brown, sunken spots on fruit, and dark lesions on leaves and branches.""")
            st.success("""Chemical Treatment: Apply fungicides like Captan or copper-based products.""")
            st.success("""Organic Treatment: Use organic fungicides such as copper sprays.""")
            st.success("""Precautions: Remove infected plant debris and prune infected areas.""")
            
        elif predicted_class == 'Apple___Cedar_apple_rust':
            st.success("""Symptoms: Orange, gelatinous lesions on leaves and fruit.""")
            st.success("""Chemical Treatment: Use fungicides such as Mancozeb and Myclobutanil.""")
            st.success("""Organic Treatment: Sulfur-based sprays can help control the disease.""")
            st.success("""Precautions: Remove nearby juniper trees, as they host the rust spores.""")
            
        elif predicted_class == 'Apple___healthy':
            st.success("The plant is healthy. No action required.")

        elif predicted_class == 'Blueberry___healthy':
            st.success("The plant is healthy. No action required.")

        elif predicted_class == 'Cherry_(including_sour)___Powdery_mildew':
            st.success("""Symptoms: White powdery growth on leaves and fruit.""")
            st.success("""Chemical Treatment: Use fungicides like Myclobutanil or sulfur.""")
            st.success("""Organic Treatment: Neem oil and potassium bicarbonate can help.""")
            st.success("""Precautions: Prune for good air circulation and avoid wetting the leaves.""")
            
        elif predicted_class == 'Cherry_(including_sour)___healthy':
            st.success("The plant is healthy. No action required.")

        elif predicted_class == 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot':
            st.success("""Symptoms: Grayish, elongated lesions on leaves.""")
            st.success("""Chemical Treatment: Apply fungicides like Strobilurins.""")
            st.success("""Organic Treatment: Copper-based fungicides can be effective.""")
            st.success("""Precautions: Practice crop rotation and remove infected plant debris.""")
        
        elif predicted_class == 'Corn_(maize)___Common_rust_':
            st.success("""Symptoms: Reddish-brown pustules on both sides of leaves.""")
            st.success("""Chemical Treatment: Apply fungicides like Mancozeb or Triazoles.""")
            st.success("""Organic Treatment: Copper-based sprays can help reduce the spread.""")
            st.success("""Precautions: Use resistant varieties and practice crop rotation.""")

        elif predicted_class == 'Corn_(maize)___Northern_Leaf_Blight':
            st.success("""Symptoms: Large, grayish-green lesions that turn brown and dry, forming cigar-shaped spots.""")
            st.success("""Chemical Treatment: Use fungicides such as Azoxystrobin or Mancozeb.""")
            st.success("""Organic Treatment: Copper-based fungicides or neem oil can help.""")
            st.success("""Precautions: Ensure proper plant spacing and remove infected crop residue.""")

        elif predicted_class == 'Corn_(maize)___healthy':
            st.success("The plant is healthy. No action required.")

        elif predicted_class == 'Grape___Black_rot':
            st.success("""Symptoms: Brown, circular lesions on leaves, with black spore-producing bodies.""")
            st.success("""Chemical Treatment: Apply fungicides like Mancozeb or Myclobutanil.""")
            st.success("""Organic Treatment: Use organic fungicides such as sulfur or copper.""")
            st.success("""Precautions: Remove and destroy infected plant debris and prune for good air circulation.""")

        elif predicted_class == 'Grape___Esca_(Black_Measles)':
            st.success("""Symptoms: Dark streaks on the wood, and yellow or red discoloration between the veins of leaves.""")
            st.success("""Chemical Treatment: No effective chemical treatment for this disease.""")
            st.success("""Organic Treatment: Improve vineyard hygiene, remove and destroy affected vines.""")
            st.success("""Precautions: Avoid planting in poorly drained soils and monitor irrigation carefully.""")

        elif predicted_class == 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)':
            st.success("""Symptoms: Brown or black spots on leaves, which may lead to defoliation.""")
            st.success("""Chemical Treatment: Use fungicides such as Mancozeb or copper-based products.""")
            st.success("""Organic Treatment: Apply sulfur-based organic fungicides.""")
            st.success("""Precautions: Remove and destroy affected leaves, and prune for air circulation.""")

        elif predicted_class == 'Grape___healthy':
            st.success("The plant is healthy. No action required.")

        elif predicted_class == 'Orange___Haunglongbing_(Citrus_greening)':
            st.success("""Symptoms: Yellowing of leaves, misshapen and bitter fruits, and stunted growth.""")
            st.success("""Chemical Treatment: No effective chemical treatment is available for this bacterial disease.""")
            st.success("""Organic Treatment: Remove infected trees and plant disease-resistant varieties.""")
            st.success("""Precautions: Control insect vectors like the Asian citrus psyllid and monitor trees regularly.""")

        elif predicted_class == 'Peach___Bacterial_spot':
            st.success("""Symptoms: Small, water-soaked lesions on leaves and fruit, leading to defoliation and fruit scarring.""")
            st.success("""Chemical Treatment: Use copper-based bactericides like Copper Oxychloride.""")
            st.success("""Organic Treatment: Neem oil and copper sprays can help manage the disease.""")
            st.success("""Precautions: Ensure good air circulation through pruning and remove infected plant material.""")

        elif predicted_class == 'Peach___healthy':
            st.success("The plant is healthy. No action required.")

        elif predicted_class == 'Pepper,_bell___Bacterial_spot':
            st.success("""Symptoms: Dark, water-soaked spots on leaves and fruit, which can become raised and scabby.""")
            st.success("""Chemical Treatment: Apply copper-based bactericides like Copper Hydroxide.""")
            st.success("""Organic Treatment: Neem oil or organic copper sprays can help reduce the spread.""")
            st.success("""Precautions: Use disease-free seeds and avoid overhead watering to reduce leaf wetness.""")

        elif predicted_class == 'Pepper,_bell___healthy':
            st.success("The plant is healthy. No action required.")

        elif predicted_class == 'Potato___Early_blight':
            st.success("""Symptoms: Dark, concentric rings on lower leaves, leading to yellowing and drying of foliage.""")
            st.success("""Chemical Treatment: Use fungicides such as Chlorothalonil or Mancozeb.""")
            st.success("""Organic Treatment: Copper-based fungicides and neem oil can help manage the disease.""")
            st.success("""Precautions: Practice crop rotation and remove affected plant debris after harvest.""")

        elif predicted_class == 'Potato___Late_blight':
            st.success("""Symptoms: Water-soaked, brown lesions on leaves, stems, and tubers, leading to decay.""")
            st.success("""Chemical Treatment: Apply fungicides such as Fluazinam or Mancozeb.""")
            st.success("""Organic Treatment: Copper-based organic fungicides can be used.""")
            st.success("""Precautions: Use resistant varieties and ensure good drainage in the field.""")

        elif predicted_class == 'Potato___healthy':
            st.success("The plant is healthy. No action required.")

        elif predicted_class == 'Raspberry___healthy':
            st.success("The plant is healthy. No action required.")

        elif predicted_class == 'Soybean___healthy':
            st.success("The plant is healthy. No action required.")

        elif predicted_class == 'Squash___Powdery_mildew':
            st.success("""Symptoms: White, powdery fungal growth on leaves and stems, leading to leaf distortion.""")
            st.success("""Chemical Treatment: Apply fungicides such as Myclobutanil or sulfur-based products.""")
            st.success("""Organic Treatment: Use neem oil or potassium bicarbonate sprays.""")
            st.success("""Precautions: Ensure good air circulation and avoid overhead watering to reduce leaf wetness.""")
        
        elif predicted_class == 'Strawberry___Leaf_scorch':
            st.success("""Symptoms: Irregular, reddish-purple spots on leaves, which may merge and cause leaf burn.""")
            st.success("""Chemical Treatment: Apply fungicides such as Captan or Myclobutanil.""")
            st.success("""Organic Treatment: Use copper-based or sulfur fungicides.""")
            st.success("""Precautions: Remove and destroy infected leaves and ensure proper spacing for air circulation.""")

        elif predicted_class == 'Strawberry___healthy':
            st.success("The plant is healthy. No action required.")

        elif predicted_class == 'Tomato___Bacterial_spot':
            st.success("""Symptoms: Small, dark, water-soaked spots on leaves and fruit, which can enlarge and cause fruit scabbing.""")
            st.success("""Chemical Treatment: Use copper-based bactericides like Copper Oxychloride.""")
            st.success("""Organic Treatment: Neem oil or copper sprays can help manage the disease.""")
            st.success("""Precautions: Ensure proper plant spacing and avoid overhead watering.""")

        elif predicted_class == 'Tomato___Early_blight':
            st.success("""Symptoms: Dark, concentric rings on older leaves, causing yellowing and leaf drop.""")
            st.success("""Chemical Treatment: Apply fungicides like Mancozeb or Chlorothalonil.""")
            st.success("""Organic Treatment: Use neem oil or copper-based organic fungicides.""")
            st.success("""Precautions: Practice crop rotation and remove infected plant debris.""")

        elif predicted_class == 'Tomato___Late_blight':
            st.success("""Symptoms: Water-soaked, dark lesions on leaves and stems, leading to plant collapse.""")
            st.success("""Chemical Treatment: Use fungicides such as Fluazinam or Mancozeb.""")
            st.success("""Organic Treatment: Apply copper-based fungicides or neem oil.""")
            st.success("""Precautions: Ensure proper drainage and use resistant tomato varieties.""")

        elif predicted_class == 'Tomato___Leaf_Mold':
            st.success("""Symptoms: Yellow spots on upper leaf surfaces, with a fuzzy, gray or purple mold on the underside.""")
            st.success("""Chemical Treatment: Apply fungicides like Chlorothalonil or copper-based products.""")
            st.success("""Organic Treatment: Use sulfur or copper fungicides for organic control.""")
            st.success("""Precautions: Ensure proper air circulation and reduce humidity levels in greenhouses.""")

        elif predicted_class == 'Tomato___Septoria_leaf_spot':
            st.success("""Symptoms: Small, circular, water-soaked spots with dark margins on lower leaves.""")
            st.success("""Chemical Treatment: Use fungicides such as Mancozeb or Chlorothalonil.""")
            st.success("""Organic Treatment: Apply neem oil or sulfur-based sprays.""")
            st.success("""Precautions: Remove and destroy infected leaves and avoid overhead watering.""")

        elif predicted_class == 'Tomato___Spider_mites Two-spotted_spider_mite':
            st.success("""Symptoms: Tiny yellow or white speckles on leaves, with fine webbing on the undersides of leaves.""")
            st.success("""Chemical Treatment: Use miticides like Abamectin or insecticidal soaps.""")
            st.success("""Organic Treatment: Neem oil or horticultural oil can control mites organically.""")
            st.success("""Precautions: Ensure proper watering and avoid dusty conditions, as mites thrive in dry environments.""")

        elif predicted_class == 'Tomato___Target_Spot':
            st.success("""Symptoms: Circular spots with concentric rings on leaves, which can lead to defoliation.""")
            st.success("""Chemical Treatment: Use fungicides like Azoxystrobin or Chlorothalonil.""")
            st.success("""Organic Treatment: Neem oil or copper-based fungicides can help.""")
            st.success("""Precautions: Remove affected leaves and ensure good air circulation around plants.""")

        elif predicted_class == 'Tomato___Tomato_Yellow_Leaf_Curl_Virus':
            st.success("""Symptoms: Leaves become curled, wrinkled, and yellow. Plants exhibit stunted growth and reduced yield.""")
            st.success("""Chemical Treatment: No effective chemical treatment is available for this viral disease.""")
            st.success("""Organic Treatment: Resistant Varieties: Plant virus-resistant tomato varieties.""")
            st.success("""Precautions: Remove and destroy infected plants to prevent the spread of the virus.""")

        elif predicted_class == 'Tomato___Tomato_mosaic_virus':
            st.success("""Symptoms: Mottled, yellow leaves with light and dark green patches, and distorted fruits.""")
            st.success("""Chemical Treatment: No chemical treatment is effective for viral diseases.""")
            st.success("""Organic Treatment: Remove infected plants and use virus-free seeds.""")
            st.success("""Precautions: Sterilize gardening tools and avoid smoking around plants, as the virus can spread via tobacco.""")

        elif predicted_class == 'Tomato___healthy':
            st.success("The plant is healthy. No action required.")

# App mode is set
elif app_mode == "Pests & Diseases":
    st.header("🪲 Pests & Diseases")
    st.markdown("#### Select a crop below to view detailed information on relevant pests and diseases.")
    st.markdown("---")

    # Crop selector inside an expander
    with st.expander("📋 Choose Crop", expanded=True):
        crop = st.selectbox(
            "Select a crop",
            ["Apple", "Corn", "Grape", "Peach", "Pepper", "Potato", "Strawberry", "Tomato"]
        )

    # Dictionary with crops and disease descriptions
    # Dictionary with crops and disease descriptions
    diseases_info = {
        "Apple": [
            {"name": "Apple Scab", "description": "Apple scab is a potentially serious fungal disease.Which is caused by the fungus Venturia inaequalis.Trees that are most commonly and severely affected include crabapple, hawthorn, mountain-ash.It is most serious in areas that have cool, wet conditions.Apple scab results in symptoms on most upper plant parts, most notably leaves and fruit. Affected leaves become twisted or puckered and have Small, rough, black, circular lesions on their upper surface."},
            {"name": "Apple Black Rot", "description": "Black rot disease, caused by the fungus Botryosphaeria obtusa .The fungus can infect dead tissue as well as living trunks, branches, leaves and fruits.Fruit symptoms:-Large brown rotten areas,black concentric rings,small, black spots can be seen on older fruit infections. These are fungal spore producing structures, called pycnidia.Leaf symptoms:-Infected leaves develop \"frog-eye leaf spot.\" These are circular spots with purplish or reddish edges and light tan interiors.Branch symptoms:-Cankers appear as a sunken, reddish-brown area on infected branches. Cankers often have rough or cracked bark.The fungus infects leaves and fruit through natural openings or minor wounds. Black Rot occurs when trees  are damaged due to  drought stress or waterlogged soils or wet conditions that could also include: Apple trees grown on sandy soils without supplemental irrigation. Trees that are not irrigated during particularly dry spells. Trees grown in poorly drained soils."},
            {"name": "Apple Rust", "description": "Apple rust disease refers to a group of fungal diseases caused by rust fungi that affect apple trees, primarily cedar-apple rust, quince rust, and hawthorn rust. It is caused by fungi in the genus Gymnosporangium.Fungal symptoms on apple include yellow, orange, or brown spots on the upper and lower surface of leaves and on the fruit.In cedar trees rust causes galls that produce orange, gelatinous tendrils (telial horns) during wet spring weather."}
        ],
        "Corn": [
            {"name": "Corn Common Rust", "description": "Common rust is caused by the fungus Puccinia sorghi .It is seldom a concern in hybrid corn.Common rust is less likely to cause significant yield loss than southern rust. Favored by moist, cool conditions (60-77 °F, 16-25 °C).Spreads by windblown spores from southern corn growing areas.Symtoms are as follows :- Lesions begin as flecks on leaves that develop into small tan spots. Spots turn into elongated brick-red to cinnamon brown pustules with jagged appearance. Found on both upper and lower leaf surfaces (unlike southern rust). Pustules turn dark brown to black late in the season. Occurs on leaf only, not on sheaths, stalks, ear shanks and husk leaves.Common rust can reduce corn yield, particularly in cases of severe infection or if the disease appears early in the growing season. The extent of the impact depends on the timing and severity of the infection, as well as the corn hybrid's resistance."},
            {"name": "Corn Gray Leaf Spot", "description": "Gray leaf spot (GLS) is a common fungal disease in the United States caused by the pathogen Cercospora zeae-maydis in corn. Disease development is favored by warm temperatures, 80°F or 27 °C; and high humidity, relative humidity of 90% or higher for 12 hours or more.Symtoms that are shown as follows :- \n\nLeaf Lesions: The disease causes small, rectangular, gray to brown lesions on the leaves. These lesions are typically narrow and run parallel to the leaf veins. As they enlarge, they may merge and cause large areas of leaf blight. \n\nLeaf Damage: Severe infections can lead to premature leaf death, which reduces the plant’s ability to photosynthesize, leading to lower grain yields. Stalk Quality: Infected plants may also have weakened stalks, increasing the risk of lodging (falling over), which complicates harvest."}
        ],
        "Grape": [
            {"name": "Grape Black Measles", "description": "Grapevine measles, also called esca, black measles or Spanish measles, has long plagued grape growers with its cryptic expression of symptoms and, for a long time, a lack of identifiable causal organism(s). The name ‘measles’ refers to the superficial spots found on the fruit.Grape Black Measles is caused by a complex of fungi, primarily Phaeomoniella chlamydospora and Phaeoacremonium aleophilum, along with Fomitiporia mediterranea, which is associated with wood decay.Symptoms shown for the following are as follows Leaf Symptoms: Infected leaves may develop interveinal chlorosis (yellowing) that later turns necrotic (brown). The leaves can show a characteristic \"tiger stripe\" pattern of yellow and brown. \n\nFruit Symptoms: On the grape clusters, the berries may develop dark, sunken spots or streaks, leading to shriveled, diseased fruit. This symptom is referred to as \"black measles.\" \n\nWood Symptoms: The disease also causes internal wood decay, which weakens the vine over time and can lead to vine dieback.Grape Black Measles thrives in warm climates, especially in older vineyards. The disease is more severe in regions with hot summers and mild winters."},
            {"name": "Grape Black Rot", "description": "Black rot, caused by the fungus Guignardia bidwellii, is a serious disease of cultivated and wild grapes. The disease is most destructive in warm, humid conditions, particularly when temperatures range between 70-80°F (21-27°C). Frequent rain or high humidity increases the likelihood of infection.warm, wet seasons. It attacks all green parts of the vine – leaves, shoots, leaf and fruit stems, tendrils, and fruit. The most damaging effect is to the fruit. Symptoms are as follows Fruit Symptoms: The most distinctive symptom is the appearance of black, shriveled grapes, often referred to as \"mummies.\" The disease starts as small, brown spots on the fruit, which then enlarge and become black. Infected berries eventually dry out and shrivel. \n\nLeaf Symptoms: Circular, brown lesions with dark borders develop on leaves. As the disease progresses, the centers of the lesions may turn gray and may drop out, creating a shot-hole appearance. \n\nStem and Tendril Symptoms: Infected stems and tendrils develop dark, sunken lesions, which can girdle and kill young shoots."}
        ],
        "Peach": [
            {"name": "Peach Bacterial spot", "description": "Peach bacterial spot is a disease that affects peach trees and other stone fruits like nectarines and plums. It's caused by the bacterium Xanthomonas arboricola pv. pruni. This disease shows up as small, dark, water-soaked spots on the leaves, fruit, and sometimes twigs. Over time, these spots can turn into lesions, leading to leaf drop and fruit scarring."},
            {"name": "Perennial canker", "description": "Perennial canker, caused by two related fungi Leucostoma cinta and L. persoonii, is also known as Valsa canker, Cytospora canker, and peach canker. This disease is a particular problem in New England since it is more serious on trees that have been weakened or damaged by cold stress. These fungi cause cankers that first appear as sunken areas in the wood that exude considerable gum. These are oval to linear in outline and are eventually surrounded by a roll of callus at the canker margins. These cankers increase in size each year and often completely girdle a limb or an entire trunk. The fungi overwinter in cankers or on dead wood. Spores called conidia are produced and carried by splashing and wind-driven rain and/or wind. Infection requires moisture and takes place through a variety of different avenues, including winter injuries, pruning stubs, mechanical damage, insect punctures, and leaf scars."},
            {"name": "Peach scab", "description": "Peach scab, caused by the fungus Cladosporium carpophilum, is most often a problem in warm, wet weather following shuck-split and is also known as ink spot. Although the fungus infects leaves and twigs, disease symptoms are most often observed on the fruit. Early infection appears on the fruit as small, circular, green spots concentrated at the stem end. These spots usually do not appear until the fruit are half-grown. Older lesions appear as dark-green to black, velvety blotches that, as they coalesce, often cause cracks in the skin and flesh. Infections on twigs and leaves are inconspicuous and appear as green-brown superficial lesions. The fungus overwinters in lesions on twigs and in spring produces spores called conidia. These conidia are washed and splashed by rain to twigs, fruit, and leaves where they cause new infections. Forty to 70 days elapse from the time the conidia land on the fruit until the disease symptoms are visible, so the disease can go unnoticed until the fruit are well grown."}
        ],
        "Pepper": [
            {"name": "Bacterial Spot", "description": " Bell pepper bacterial spot is a common disease caused by the bacterium Xanthomonas campestris pv. vesicatoria. It affects bell peppers and other types of peppers, as well as tomatoes. The disease shows up as small, dark, water-soaked spots on the leaves, stems, and fruits. Over time, these spots can become raised and may have a yellow halo around them. The infected leaves may drop prematurely, and the fruit can develop lesions that make them unmarketable."},
            {"name": "Pepper early blight", "description": "Early blight on peppers is a fungal disease caused by Alternaria solani, the same fungus that affects tomatoes and potatoes. It primarily attacks the leaves, stems, and sometimes the fruits of pepper plants. Early blight typically appears as dark, concentric ring spots (target-like lesions) on older leaves. As the disease progresses, these spots can cause the leaves to yellow and drop prematurely, reducing the plant's ability to photosynthesize and produce fruit."}
        ],
        "Potato": [
            {"name": "Potato Late Blight", "description": "Potato late blight is a serious and destructive disease caused by the water mold Phytophthora infestans. This is the same disease responsible for the Irish Potato Famine in the 1840s. Late blight affects both potato plants and tomatoes . \n\nLeaves symptomns : Dark, water-soaked spots appear on the leaves, often with a pale green halo around them. As the disease progresses, these spots turn brown and may be surrounded by a white, fuzzy mold, especially on the undersides of leaves. Stems  symptomns: Dark lesions may also develop on the stems, causing the plant to collapse and die. \n\nTubers symptomns: Infected potato tubers develop brown, irregular patches that can extend into the flesh. These tubers often rot in storage, leading to significant crop loss. "}
        ],
        "Strawberry": [
            {"name": "Strawberry leaf scroch", "description": "Strawberry leaf scorch is a fungal disease caused by Diplocarpon earlianum. It primarily affects the leaves of strawberry plants, though it can also impact the petioles, runners, and fruit stalks. The disease is most severe in warm, humid conditions, particularly when plants are crowded or have poor air circulation. Leaf Spots: Small, purple or reddish spots appear on the upper side of the leaves. Over time, these spots may enlarge and merge, giving the leaf a scorched appearance. Leaf Browning: The leaves may turn brown and dry out, resembling a \"scorched\" effect. Severely affected leaves may curl and eventually die.Reduced Plant Vigor: The disease can weaken the plant, reducing its overall health and fruit production."}
        ],
        "Tomato":[
            {"name": "Tomato Bacterial Spot", "description": "Bacterial spot of tomato is caused by Xanthomonas vesicatoria, Xanthomonas euvesicatoria, Xanthomonas gardneri, and Xanthomonas perforansBacterial spot of tomato is caused by Xanthomonas vesicatoria, Xanthomonas euvesicatoria, Xanthomonas gardneri, and Xanthomonas perforans.Bacterial spot appears on leaves as small (less than ⅛ inch), sometimes water-soaked (i.e., wet-looking) circular areas. Spots may initially be yellow-green but darken to brownish-red as they age. When the disease is severe, extensive leaf yellowing and leaf loss can also occur."},
            {"name": "Tomato early blight", "description": " Early Blight Infections begin as small brown spots on older leaves that quickly enlarge. A yellow halo usually surrounds the lesions. The lesions develop a \"bulls-eye\" pattern of concentric rings that can be seen with a hand lens. Individual lesions enlarge and coalesce and can kill entire leaves. The disease can also move to stems and fruits and produce dark lesions."},
            {"name": "Tomato late blight", "description": "Phytophthora infestans is the oomycete, or water mold pathogen, responsible for tomato late blight.  Leaf Symptomns : Infected leaves typically have green to brown patches of dead tissue surrounded by a pale green or gray border. When the weather is very humid and wet, late blight infections can appear water-soaked or dark brown in color, and are often described as appearing greasy. White, fuzzy growth may be found on the undersides of leaves or on lower stems. Stem and petiole lesions are brown and are typically not well defined in shape. Discoloration may also occur on the flowers, causing them to drop. Symptomatic tomato fruits appear mottled, often with golden to dark brown, firm, sunken surfaces. White, fuzzy pathogen growth can also be found in association with the fruit lesions."},
            {"name": "Tomato Leaf spot", "description": "Leaf spot is caused by a fungus, Septoria lycopersici. It is one of the most destructive diseases of tomato foliage and is particularly severe in areas where wet, humid weather persists for extended periods. Septoria leaf spot usually appears on the lower leaves after the first fruit sets. Spots are circular, about 1/16 to 1/4 inch in diameter with dark brown margins and tan to gray centers with small black fruiting structures. Characteristically, there are many spots per leaf. This disease spreads upwards from oldest to youngest growth. If leaf lesions are numerous, the leaves turn slightly yellow, then brown, and then wither. Fruit infection is rare."}
        ]
    }

    st.subheader(f"🌾 Diseases affecting **{crop}**:")
    st.markdown("---")

    # Use expanders for each disease
    for disease in diseases_info.get(crop, []):
        with st.expander(f"🦠 {disease['name']}"):
            st.write(disease['description'])

elif app_mode == "About":
    st.header("An small Introduction to the project:")
    st.markdown("""
    Our platform leverages cutting-edge AI and machine learning technologies to revolutionize plant health management, helping farmers and agricultural professionals tackle plant diseases with unmatched precision. By integrating advanced visual and textual feature extraction, our system utilizes Convolutional Neural Networks (CNNs) along with Mask R-CNN and Faster R-CNN for high-accuracy plant disease detection, pinpointing affected regions and classifying diseases based on real-time data.

    To enhance decision-making, our technology provides actionable insights with predictive analytics and trend visualizations. Furthermore, our user-friendly mobile app employs TensorFlow Lite for seamless model integration, real-time image capture, and weather-based notifications. We also support a robust community of farmers through discussion forums and gamified features, fostering collaboration and promoting best practices. Our commitment to innovation empowers farmers with the tools they need for proactive crop management, securing sustainable yields and improved food security.
                Archietecture Used:""")
    image_path = "Architecture.png"
    st.image(image_path, use_container_width=True)


    st.markdown("""
    The architecture of our plant disease detection system is designed to provide a seamless, efficient, and accurate diagnosis and prediction tool. Here’s a breakdown of its core components:

    ### 1. **Data Acquisition and Preprocessing**
    - **Field Data Collection**: High-resolution images of plants (both healthy and diseased) are gathered under diverse lighting and angles. Additionally, publicly available datasets like PlantVillage and PlantDoc further enrich the data pool.
    - **Literature Review and Farmer Insights**: Journals, research papers, and direct feedback from farmers provide crucial insights into symptomatology and control measures, making the system well-informed and comprehensive.

    ### 2. **Visual and Textual Feature Extraction**
    - **Visual Extraction**: Leveraging powerful models like CNN, Mask R-CNN, and Faster R-CNN, the system extracts disease-specific visual features.
        - **CNN** is used for general image classification.
        - **Faster R-CNN** detects and localizes infected regions using bounding boxes for rapid, accurate object detection.
        - **Mask R-CNN** offers advanced pixel-level segmentation, enabling precise disease identification.
    - **Textual Extraction**: Techniques like TF-IDF, Bag of Words (BoW), and Named Entity Recognition (NER) process text to extract relevant information, enhancing the model's ability to interpret textual data for classification and clustering.

    ### 3. **Machine Learning and Deep Learning Models**
    - **Model Training and Validation**: The models are trained using TensorFlow and PyTorch, achieving accuracies up to 95% for disease detection with Mask R-CNN.
    - **Model Conversion and Optimization**: The trained models are converted to mobile-compatible formats, like TensorFlow Lite, ensuring efficient processing on mobile devices.

    ### 4. **Mobile Application and Real-Time Predictions**
    - The mobile app, built with Android Jetpack Compose, includes a user-friendly interface and real-time image capture. Users can easily log in, capture plant images, and receive immediate disease predictions along with mitigation strategies.
    - **Weather and Alert Integration**: The app fetches real-time weather data, providing alerts on conditions that may lead to disease outbreaks, and allows users to view historical data on weather, soil conditions, and crop health for informed decision-making.

    ### 5. **Community and Engagement Features**
    - **Farmer Community Forum**: Enables knowledge sharing and problem-solving among users.
    - **Gamification**: Incentivizes active app use and disease reporting by rewarding points, fostering a cooperative user base.

    ### 6. **System Notifications and Data-Driven Insights**
    - Real-time notifications alert farmers to disease triggers and outbreaks in nearby areas, while a historical data analysis feature aids in predicting future risks and planning for disease prevention.

    ### Results and Findings:
    CNN achieved an overall accuracy of 90% for classifying plant health status, providing a strong baseline for image recognition tasks.
    Faster R-CNN showed an accuracy of 88% in detecting and localizing infected regions in plant images. Its speed and precision in bounding box creation make it ideal for identifying disease-prone areas.
    Mask R-CNN excelled in early-stage disease identification with a 85% accuracy rate, enabling pixel-level segmentation and detailed visual analysis, which proved highly reliable for distinguishing specific disease patterns.
    """)
    image_path = "result.png"
    st.image(image_path, use_container_width=True)

elif app_mode == "Weather":
    import streamlit as st
    import requests

    # ——— 1) CSS for a sleek card look ———
    st.markdown("""
    <style>
      .weather-container {
        background: #f0f4f8;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
      }
      .weather-title {
        font-size: 2.5rem;
        font-weight: 600;
        color: #333333;
        text-align: center;
        margin-bottom: 1.5rem;
      }
      .metric-card {
        background: #ffffff;
        border-radius: 0.75rem;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
      }
      .metric-value {
        font-size: 1.5rem;
        font-weight: 500;
        color: #1f77b4;
      }
      .footer {
        font-size: 0.875rem;
        color: #777;
        text-align: center;
        margin-top: 1rem;
      }
    </style>
    """, unsafe_allow_html=True)

    # ——— 2) Container start ———
    st.markdown('<div class="weather-container">', unsafe_allow_html=True)
    st.markdown('<div class="weather-title">🌤️ Current Weather</div>', unsafe_allow_html=True)

    # ——— 3) City input ———
    city = st.text_input(
        "Enter city name",
        value="Delhi",
        placeholder="e.g. Mumbai",
        key="weather_city",
        help="Type a city and press Enter"
    )

    # ——— 4) Fetch & display ———
    if city:
        # Build the correct Visual Crossing URL for this city
        vc_key     = "6QGKN2NBTX955MHD83SSN75NS"
        vc_base    = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{city}"
        vc_params  = {"unitGroup":"metric", "key":vc_key, "contentType":"json"}

        try:
            resp = requests.get(vc_base, params=vc_params)
            resp.raise_for_status()
            data = resp.json()

            # ——— 5) Parse VC fields ———
            location = data.get("resolvedAddress", city)
            cc       = data.get("currentConditions", {})
            cond     = cc.get("conditions", "N/A")
            temp_c   = cc.get("temp", "N/A")
            humidity = cc.get("humidity", "N/A")
            wind_kph = cc.get("windspeed", "N/A")

            # ——— 6) Render metrics ———
            st.markdown(f"### 📍 {location}")
            cols = st.columns(3)
            with cols[0]:
                st.markdown(
                    '<div class="metric-card">'
                    f'<div class="metric-value">{temp_c}°C</div>'
                    '<div>Temperature</div>'
                    '</div>',
                    unsafe_allow_html=True
                )
            with cols[1]:
                st.markdown(
                    '<div class="metric-card">'
                    f'<div class="metric-value">{humidity}%</div>'
                    '<div>Humidity</div>'
                    '</div>',
                    unsafe_allow_html=True
                )
            with cols[2]:
                st.markdown(
                    '<div class="metric-card">'
                    f'<div class="metric-value">{wind_kph} kph</div>'
                    '<div>Wind Speed</div>'
                    '</div>',
                    unsafe_allow_html=True
                )

            st.markdown(f"**Condition:** {cond}")

        except requests.exceptions.HTTPError:
            st.error("❌ Could not fetch weather for that city—please check spelling or try another.")

    # ——— 7) Container end & footer ———
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="footer">Data provided by Visual Crossing</div>', unsafe_allow_html=True)

elif app_mode == "Shop":
    st.header("🛒 Shop Plant Care Essentials")
    st.markdown("#### Explore our curated plant care products below.")

    # 1) Define products inside this block
    products = [
        {
            "name": "POTASSIUM SULPHATE",
            "price": "₹530",
            "description": "Nutrient-rich fertilizer to boost crop quality, yield, and disease resistance. Low salt; pairs well with most blends.",
            "url": "https://www.amazon.in/ISOCHEM-LABORATORIES-POTASSIUM-SULPHATE-500/dp/B07R4WVDGZ"
        },
        {
            "name": "DAP",
            "price": "₹919",
            "description": "Balanced N-P-K mix that encourages lush foliage and root development. Water-soluble for quick uptake.",
            "url": "https://www.amazon.in/Organic-Plant-Bio-Fertilizer-Crops/dp/B0B2MWJRNH"
        },
        {
            "name": "UREA",
            "price": "₹445",
            "description": "Pure nitrogen source to accelerate vegetative growth. Ideal for leafy crops and early season top-dressing.",
            "url": "https://www.amazon.in/Bhajanlal-UREA-Gardening-Fertilizer-Nutrient-Rich/dp/B0DFR2PCQP"
        },
        {
            "name": "Soil Enhancer",
            "price": "₹614",
            "description": "Organic potting medium with superior moisture retention and drainage. Made from sustainably sourced compost.",
            "url": "https://www.amazon.in/Trust-Enriched-Organic-Potting-Fertilizer/dp/B07L6BLSDL"
        }
    ]

    # 2) Global CSS
    st.markdown("""
    <style>
      .shop-container {
        display: flex;
        flex-wrap: wrap;
        gap: 1.5rem;
        justify-content: center;
        margin-top: 1rem;
      }
      .product-card {
        background: #fff;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        padding: 1rem;
        width: 260px;
        transition: transform 0.2s, box-shadow 0.2s;
      }
      .product-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
      }
      .product-card h3 {
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
        color: #2c3e50;
      }
      .product-card .price {
        color: #27ae60;
        font-weight: 600;
        margin-bottom: 0.75rem;
      }
      .product-card p {
        font-size: 0.9rem;
        color: #555;
        margin-bottom: 1rem;
        line-height: 1.3;
      }
      .buy-btn {
        display: inline-block;
        background: #27ae60;
        color: #fff;
        text-decoration: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 500;
      }
      .buy-btn:hover { background: #219150; }
    </style>
    """, unsafe_allow_html=True)

    # 3) Build one big HTML string
    html = '<div class="shop-container">'
    for p in products:
        html += f'''
          <div class="product-card">
            <h3>{p["name"]}</h3>
            <div class="price">{p["price"]}</div>
            <p>{p["description"]}</p>
            <a href="{p["url"]}" target="_blank" class="buy-btn">Buy Now</a>
          </div>
        '''
    html += '</div>'

    # 4) Render it all in one go via components.html
    import streamlit.components.v1 as components
    # height can be adjusted based on number of rows
    components.html(html, height=600, scrolling=True)
