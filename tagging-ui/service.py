import os
import io
import typing as tp

import requests
import streamlit as st

PAGE_TITLE = "[REMY] Tagging UI"
IMAGE_TAGGING_ENDPOINT = os.getenv("IMAGE_TAGGING_ENDPOINT", "http://localhost:3000/predict")

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="🖼️"    
)

def clear_tagging_result():
    if "tagging_result" in st.session_state:
        st.session_state.pop("tagging_result")
    assert "tagging_result" not in st.session_state

def infer_image(image_bytes_stream: io.BytesIO, extension: str = 'jpeg', endpoint_url: str = IMAGE_TAGGING_ENDPOINT):
    if extension == 'jpg':
        extension = 'jpeg'
    headers = {'accept': 'application/json'}
    files = {'imgs': (f'inference_image.{extension}', image_bytes_stream, f'image/{extension}')}
    response = requests.post(endpoint_url, headers=headers, files=files)
    if response.status_code != 200:
        st.error(f"Server has failed to process request. Status code: {response.status_code}. Response content: {response.content}")
        return
    st.session_state['tagging_result'] = sorted(response.json()[0].get("tags", []))
    st.toast("An image has been tagged successfully!", icon='🎉')


def main_page():
    st.header(PAGE_TITLE)
    st.text("A simple service for ML-based tagging of images")
    image = st.file_uploader(
        "Upload an image for processing", 
        type=['jpg', 'png'],
        on_change=clear_tagging_result
    )
    if image:
        st.markdown("Provided image:")
        st.image(image)
    btn = st.button("Submit for tagging", disabled=not image)
    send_to_inference_condition = btn and image and not st.session_state.get("tagging_result")
    
    if not send_to_inference_condition:
        return
    
    with st.spinner("Please wait for image to be processed..."):
        image_stream = io.BytesIO(image.getvalue())
        image_extension = image.name.split('.')[-1].lower()
        infer_image(image_stream, image_extension)

    if st.session_state.get("tagging_result"):
        tags: tp.List[str] = st.session_state.get('tagging_result')
        tags = [t.replace("_", " ").strip().capitalize() for t in tags]
        tagging_result_list = '\n\n- '.join(tags)
        st.markdown(f"Result tags:\n- {tagging_result_list}")
        


    

if __name__ == '__main__':
    main_page()