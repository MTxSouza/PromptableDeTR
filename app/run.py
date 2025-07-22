"""
Script used to run a browser application to use the PromptableDeTR model.
"""
import time

import streamlit as st
from PIL import Image


# Widgets.
@st.fragment(run_every=10)
def custom_title():
    title_widget = st.empty()
    html_func = lambda init, final: f"""
<style>
    .initTitle {{
        font-size: 60px;
        font-weight: bold;
    }}
    .finalTitle {{
        font-size: 60px;
        font-weight: bold;
        color: #F76C54;
    }}
</style>
<div style="text-align: center;">
    <span class="initTitle">{init}</span><span class="finalTitle">{final}</span>
</div>
"""

    init_title = []
    for char in "Promptable":
        init_title.append(char)
        html = html_func("".join(init_title), "")
        title_widget.markdown(html, unsafe_allow_html=True)
        time.sleep(0.05)
    init_title = "".join(init_title)

    final_title = []
    delete = False
    for _ in range(2):
        for char in "det":
            if delete:
                final_title.pop()
            else:
                final_title.append(char)

            html = html_func(init_title, "".join(final_title))
            title_widget.markdown(html, unsafe_allow_html=True)
            time.sleep(0.1)
        delete = not delete

    for char in "DeTR":
        final_title.append(char)
        html = html_func(init_title, "".join(final_title))
        title_widget.markdown(html, unsafe_allow_html=True)
        time.sleep(0.12)


def sidebar_widget():

    # Image upload widget.
    with st.sidebar:
        image_filepath = st.file_uploader(label="Image", type=[".jpg", ".jpeg", ".png"], help="Path to the image file.")
        st.session_state["image"] = image_filepath

@st.cache_resource()
def get_model(model_weights):
    """
    Load the model with the given weights.

    Args:
        model_weights (str): Path to the model weights file.

    Returns:
        str: A message indicating the model has been loaded.
    """
    if model_weights is None:
        return "Please upload a model weights file."

    # Here you would load your model using the provided weights.
    # For demonstration, we just return a success message.
    return f"Model loaded successfully with weights from {model_weights.name}."


def loaded_image():
    """
    Get the current image from the Streamlit session state.
    """
    return st.session_state.get("image", None)


def get_predictions():
    """
    Get the predictions from the model based on the uploaded image and input text.
    """
    return st.session_state.get("predict", None)


def load_image(image_file):
    """
    Load the image from the uploaded file.

    Args:
        image_file (UploadedFile): The uploaded image file.

    Returns:
        PIL.Image.Image: The loaded image in RGB format.
    """
    with Image.open(fp=image_file, mode="r") as img:
        img = img.convert("RGB")
        img = img.resize(size=(640, 640))
        return img


if __name__=="__main__":

    # Define title.
    custom_title()

    # Define sidebar widget.
    sidebar_widget()

    # Image viewer.
    for _ in range(5):
        st.text("")
    with st.container(border=True):

        image_filepath = loaded_image()
        if image_filepath is None:
            st.warning("Please upload an image file to view.")
        else:

            # Load and display the image.
            img = load_image(image_file=image_filepath)

            # Check if there is any predictions.
            predictions = get_predictions()
            if predictions is not None:
                pass

            st.image(img, caption="Uploaded Image", use_container_width=True)

            # Input text.
            target_text = st.text_input(label="Prompt", max_chars=50)
            if st.button(label="Detect"):
                if target_text:
                    st.rerun()
                else:
                    st.error("Please enter a prompt text.")
