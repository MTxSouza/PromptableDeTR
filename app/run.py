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

    # Model configuration.
    weight_filepath = None
    with st.expander("Model configuration"):

        # Model weights.
        weight_filepath = st.file_uploader(label="Model weights", type=[".pt", ".pth"], help="Path to the model weights file.")
        if weight_filepath is not None:
            with st.form(key="model_form"):
                prompt = st.text_input(label="Prompt", help="Description of the target object to be detected.").strip().lower()
                presence_threashold = st.slider(label="Presence threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
                if st.form_submit_button(label="Set config"):
                    if not prompt:
                        st.error("Please provide a prompt.")

    if weight_filepath is not None:
        st.divider()
        image_filepath = st.file_uploader(label="Image", type=[".jpg", ".jpeg", ".png"], help="Path to the image file.")
        if image_filepath is not None:
            img = load_image(image_file=image_filepath)
            st.image(img, caption="Uploaded Image", use_container_width=True)
