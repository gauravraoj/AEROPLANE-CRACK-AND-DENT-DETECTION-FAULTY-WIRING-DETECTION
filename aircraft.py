import streamlit as st
from roboflow import Roboflow
import supervision as sv
import cv2
import numpy as np
import json
import pickle
import pandas as pd
import tempfile
import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as gen_ai

# Load environment variables
load_dotenv()
# st.session_state.clicked = True
# Configure Streamlit page settings
st.set_page_config(
    page_title="AIRSAFETY DETECTION ",
    page_icon="	:small_airplane:",  # Favicon emoji
    layout="centered",  # Page layout option
)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# Set up Google Gemini-Pro AI model
gen_ai.configure(api_key="AIzaSyBb9YYJkW_5EyBtSsJ1QrzdLZCaoR3U5gs")
gemini_model = gen_ai.GenerativeModel('gemini-pro')

# Function to translate roles between Gemini-Pro and Streamlit terminology
def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role
# def set_background_color(color):
#   """Sets the background color of the Streamlit app."""
#   st.write(f"<style>.root {{ background-color: {color}; }}</style>", unsafe_allow_html=True)

# # Example usage (uncomment to use)
# set_background_color("#ADD8D6")  #

def custom_header(text, style="font-size: 36px; font-weight: bold;"):
    """Creates a custom header with CSS styling."""
    st.write(f"<h1 style='{style}'>{text}</h1>", unsafe_allow_html=True)
def count_classes_from_json(json_data):
  data_dict = json.loads(json_data)
  # Initialize an empty dictionary to store class counts
  class_counts = {} 
  
 
# Loop through predictions and count class occurrences
  for prediction in data_dict["predictions"]:
    # Get the class name
    class_name = prediction["class"]

    # Check if class already exists in the dictionary
    if class_name in class_counts:
        class_counts[class_name] += 1
    else:
        class_counts[class_name] = 1

# Print the class counts
  print("Class counts:")
  for class_name, count in class_counts.items():
    print(f"- {class_name}: {count}")

  return class_counts
def click_button():
    st.session_state.clicked = True
    
def click_button_1():
    st.session_state.clicked = False
    

custom_header("AIRCRAFT SAFETY ")
st.header('CRACK AND DENT :blue[ELECTRICAL FAULT]:airplane_arriving:')

col1, col2 = st.columns([1,1])  # Adjust column ratios as needed

with col1:
    if st.button("DETECTION", type="primary",on_click=click_button):
        st.write("Crack and dent detection")

with col2:
    if st.button("ELECTRICAL FAULT", type="primary",on_click=click_button_1):
        st.write("Electrical fault Detection")


if st.session_state.clicked:
    # The message and nested widget will remain on the page
    
    uploaded_file = st.file_uploader("Choose an Image (jpg or png)", type=['jpg', 'png'])
    submit_button = st.button("Submit Image",type="primary")
    if submit_button:
        # Get image data directly from uploader
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            image_path = temp_file.name  # Get the temporary file path

        rf = Roboflow(api_key="mydcymcSm1nXromS26eO")
        project = rf.workspace().project("gaurav_rao_j")
        model = project.version(2).model

        result = model.predict(image_path, confidence=2, overlap=30).json()

        labels = [item["class"] for item in result["predictions"]]

        detections = sv.Detections.from_roboflow(result)

        label_annotator = sv.LabelAnnotator()
        bounding_box_annotator = sv.BoundingBoxAnnotator()

        image = cv2.imread(image_path)

        annotated_image = bounding_box_annotator.annotate(
        scene=image, detections=detections)
        annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels)
        st.image(annotated_image, caption="DETECTED IMAGE")

        class_counts = count_classes_from_json(json.dumps(result))
        st.title("CRACK OR DENT DETECTED")
        for class_name, count in class_counts.items():
           st.header(f"- {class_name}: {count}")
        # Display the chatbot's title on the page
        st.title("SEVERITY")
        prompt=f"There are{result} in the plane during scanning.Just predict severity of the damage in one word "
        response = gemini_model.generate_content(prompt)
        st.header(response.text)

           
else:
       # Create six text boxes with labels
        text_boxes = [
        st.text_input(" Current in A"),
        st.text_input(" Current in B"),
        st.text_input(" Current in C"),
        st.text_input(" Voltage in A"),
        st.text_input(" Voltage in B"),
        st.text_input(" Voltage in C"),
                ]
        text_labels =  ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc'] 
        # Display entered text (optional)
        if all(text_boxes):  # Check if all text boxes have input
            
            detect_button=st.button("DETECT THE FAULT", type="primary")
            if detect_button:
                # Load the pickled model
                pickled_model = pickle.load(open('model.pkl', 'rb'))

                # Define the mapping between text labels and integer values
                label_mapping = {
    0: 'NO Fault',
    1: 'Line A to Ground Fault',
    2: 'Line B to Line C Fault',
    3: 'Line A Line B to Ground Fault',
    4: 'Line A Line B Line C',
    5: 'Line A Line B Line C to Ground Fault'
}
                data ={}
                for i, text_input in enumerate(text_boxes):
                # Extract the value and convert to float (assuming numerical input)
                    try:
                        data[text_labels[i]] = float(text_input.strip())
                    except ValueError:
                        st.error(f"Invalid input for Measurement {i+1}. Please enter numbers.")
                        continue 
                input_df = pd.DataFrame.from_dict([data])
                predicted_fault_type = pickled_model.predict(input_df)
                result = label_mapping[predicted_fault_type[0]]
                st.title(f"{result}")
                



                
                


       
    
        
        
