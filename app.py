import streamlit as st
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
from diabetic import diabetic_retinopathy
from redness import redi
import time

def main():
    st.title("OneMed: Tensorflow Keras based medical tools")
    st.write("------------------------------------------")
    st.sidebar.title("Command Bar")
    choices = ["Home","EyeMed", "COVID Med", "Skin Med"]
    menu = st.sidebar.selectbox("Menu: ", choices)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    if menu =="Home":
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i in range(100):
            progress_bar.progress(i + 1)
            status_text.text("Setting up the magic...")
        time.sleep(2)
        status_text.success("All Set!")
        st.write("---------------------------------")
        st.write("OneMed contains 3 main sections: Explore the sections in the menu on the sidebar. Once you select a section, you'll be asked to upload an image. Once uploaded, buttons will pop-up with function calls to the models. The results will be shown on the same page.")
    elif menu == "EyeMed":
        st.sidebar.write("EyeMed analyzes cataract, diabetic retinopathy and redness levels. Upload an image to get started.")
        st.write("---------------------------")
        image_input = st.sidebar.file_uploader("Choose an eye image: ", type="jpg")
        if image_input:
            st.sidebar.image(image_input, width=300, height=300)

            detect = st.sidebar.button("Detect Cataract")
            dr = st.sidebar.button("Analyze Diabetic Retinopathy")
            r = st.sidebar.button("Analyze Redness Levels")
            size = st.slider("Adjust Image Size: ", 300, 1000)
            st.image(image_input, width=size, height=size)
            st.write("------------------------------------------------------")
            

    # Disable scientific notation for clarity
            np.set_printoptions(suppress=True)

    # Load the model
            model = tensorflow.keras.models.load_model('eye_models/cataract/model.h5')

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
            image = Image.open(image_input)

    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
            image_array = np.asarray(image)

    # display the resized image

    # Normalize the image
            normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
            data[0] = normalized_image_array

            

            if detect:
                prediction = model.predict(data)
                class1 = prediction[0,0]
                class2 = prediction[0,1]
                if class1 > class2:
                    st.markdown("EyeMed thinks this is a **Cataract** by " + str(class1 * 100) + "%" )
                elif class2 > class1:
                    st.markdown("EyeMed thinks this is **Uveitis** by " + str(class2 * 100) + "%")
                else:
                    st.write("We encountered an ERROR. This should be temporary, please try again with a better quality image. Cheers!")

            if dr:
                answer = diabetic_retinopathy(image_input)
                class1 = answer[0,0]
                class2 = answer[0,1]
                if class1 > class2:
                    st.write("Diabetic Retinopathy Detected. Confidence: " + str(class1*100) + "%")
                    st.write("-------------------------------")
                elif class2 > class1:
                    st.write("Diabetic Retinopathy Not Detected.")
                    st.write("-------------------------------")
            if r:
                answer = redi(image_input)
                class1 = answer[0,0]
                class2 = answer[0,1]
                if class1 > class2:
                    st.write("Redness Levels: " + str(class1*100) + "%")
                    st.write("-------------------------------")
                elif class2 > class1:
                    st.write("No Redness Detected. Confidence: " + str(class1*100) + "%")
                    st.write("-------------------------------")

    elif menu == "COVID Med":
        st.sidebar.write("COVID Med uses CT Scans to detect whether the patient is likely to have COVID or not. Upload an image to get started.")
        st.write("---------------------------")
        st.set_option('deprecation.showfileUploaderEncoding', False)
        image_input = st.sidebar.file_uploader("Choose a file: ", type='png')
        if image_input:
            analyze = st.sidebar.button("Analyze")
            size = st.slider("Adjust image size: ", 300, 1000)
            st.image(image_input, width=size, height=size)
            st.write("-----------------------------------------")
            # Disable scientific notation for clarity 
            np.set_printoptions(suppress=True)
            model = tensorflow.keras.models.load_model('covid_model/38/model.h5')

        # Create the array of the right shape to feed into the keras model
        # The 'length' or number of images you can put into the array is
        # determined by the first position in the shape tuple, in this case 1.
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # Replace this with the path to your image
            if analyze: 
                image = Image.open(image_input)
            #resize the image to a 224x224 with the same strategy as in TM2:
            # #resizing the image to be at least 224x224 and then cropping from the center
                size = (224, 224)
                image = ImageOps.fit(image, size, Image.ANTIALIAS)
            # #turn the image into a numpy array
                image_array = np.asarray(image)
            # display the resized image
            # Normalize the image
                normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
            # Load the image into the array
                data[0] = normalized_image_array
            # run the inference
                prediction = model.predict(data)
                print(prediction)
                class1 = prediction[0,0]
                class2 = prediction[0,1]
                if class1 - class2 > 0.3:
                    st.markdown("**Possibility of COVID.** Confidence: " + str(class1 * 100) + "%")
                elif class2 - class1 > 0.3:
                    st.markdown("**Unlikely to have COVID**")
                else:
                    st.write("Error! Please upload a better quality image for accuracy.")
                    
    elif menu == "Skin Med":
        st.sidebar.write("Skin Med detects whether the patient has benign or malignant type of cancer. Further classifications are still under testing. Upload an image to get started.")
        st.write("---------------------------")
        st.set_option('deprecation.showfileUploaderEncoding', False)
        image_input = st.sidebar.file_uploader("Choose a file: ", type='jpg')
        if image_input:
            analyze = st.sidebar.button("Analyze")
            size = st.slider("Adjust image size: ", 300, 1000)
            st.image(image_input, width=size, height=size)
            st.write("-----------------------------------------")
            # Disable scientific notation for clarity 
            np.set_printoptions(suppress=True)
            model = tensorflow.keras.models.load_model('skin_model/model.h5')

        # Create the array of the right shape to feed into the keras model
        # The 'length' or number of images you can put into the array is
        # determined by the first position in the shape tuple, in this case 1.
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # Replace this with the path to your image
            if analyze: 
                image = Image.open(image_input)
            #resize the image to a 224x224 with the same strategy as in TM2:
            # #resizing the image to be at least 224x224 and then cropping from the center
                size = (224, 224)
                image = ImageOps.fit(image, size, Image.ANTIALIAS)
            # #turn the image into a numpy array
                image_array = np.asarray(image)
            # display the resized image
            # Normalize the image
                normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
            # Load the image into the array
                data[0] = normalized_image_array
            # run the inference
                prediction = model.predict(data)
                class1 = prediction[0,0]
                class2 = prediction[0,1]
                if class1 - class2 > 0.5:
                    st.markdown("**Benign Detected.** Confidence: " + str(class1 * 100) + "%")
                elif class2 - class1 > 0.5:
                    st.markdown("**Malign Detected.** Confidence: " + str(class2 * 100) + "%")
                else:
                    st.write("Error! Please upload a better quality image for accuracy.")


if __name__ == '__main__':
    main()
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
