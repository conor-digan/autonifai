from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from pytesseract import image_to_string
from PIL import Image
from io import BytesIO

import pytesseract
import pypdfium2 as pdfium
import streamlit as st
import multiprocessing
from tempfile import NamedTemporaryFile
import pandas as pd
import json
import os

load_dotenv()

# Set tesseract path
# Set the tesseract path
# pytesseract.pytesseract.tesseract_cmd = '/app/.apt/usr/bin/tesseract'

# Set the tessdata path to be the manually added files
os.environ['TESSDATA_PREFIX'] = './tessdata'

# 1. Convert PDF file into images via pypdfium2
def convert_pdf_to_images(file_path, scale=300/72):

    pdf_file = pdfium.PdfDocument(file_path)

    page_indices = [i for i in range(len(pdf_file))]

    renderer = pdf_file.render(
        pdfium.PdfBitmap.to_pil,
        page_indices=page_indices,
        scale=scale,
    )

    final_images = []

    for i, image in zip(page_indices, renderer):

        image_byte_array = BytesIO()
        image.save(image_byte_array, format='jpeg', optimize=True)
        image_byte_array = image_byte_array.getvalue()
        final_images.append(dict({i: image_byte_array}))

    return final_images

# 2. Extract text from images via pytesseract
def extract_text_from_img(list_dict_final_images):

    image_list = [list(data.values())[0] for data in list_dict_final_images]
    image_content = []

    for index, image_bytes in enumerate(image_list):

        image = Image.open(BytesIO(image_bytes))
        raw_text = str(image_to_string(image))
        image_content.append(raw_text)

    return "\n".join(image_content)


def extract_content_from_url(url: str):
    images_list = convert_pdf_to_images(url)
    text_with_pytesseract = extract_text_from_img(images_list)

    return text_with_pytesseract


# 3. Extract structured info from text via LLM
def extract_structured_data(content: str, data_points):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
    template = """
    You are an expert admin person who will extract core information from documents

    {content}

    Above is the content; please try to extract all data points from the content above 
    and export in a JSON array format:
    {data_points}

    Now please extract details from the content  and export in a JSON array format, 
    return ONLY the JSON array:
    """

    prompt = PromptTemplate(
        input_variables=["content", "data_points"],
        template=template,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    results = chain.run(content=content, data_points=data_points)

    return results


# # 4. Send data to make.com via webhook
# def send_to_make(data):
#     # Replace with your own link
#     webhook_url = "https://hook.eu1.make.com/xxxxxxxxxxxxxxxxx"
#
#     json = {
#         "data": data
#     }
#
#     try:
#         response = requests.post(webhook_url, json=json)
#         response.raise_for_status()  # Check for any HTTP errors
#         print("Data sent successfully!")
#     except requests.exceptions.RequestException as e:
#         print(f"Failed to send data: {e}")


# 5. Streamlit app
def main():
    data_points = """{
        "company_name": "What is the name of the company that sent the invoice?",
        "company_address": "What is the address of the company that sent the invoice?",
        "description": "What is the description of the items on the invoice?",
        "due_date": "When is the payment due on the invoice?",
        "amount": "What is the total amount due (inc. tax) on the invoice?",
        "tax": "What is the tax percentage for the invoice?",
    }"""

    st.set_page_config(page_title="Invoice data extraction", page_icon=":microscope:")

    st.header("Invoice data extraction :microscope:")

    # data_points = st.text_area(
    #     "Data points", value=default_data_points, height=170)

    uploaded_files = st.file_uploader(
        "upload PDFs", accept_multiple_files=True)

    if uploaded_files is not None:
        results = []
        print(len(uploaded_files))
        for file in uploaded_files:
            print('Filename: {0}'.format(file.name))
            with NamedTemporaryFile(dir='.', suffix='.csv') as f:
                f.write(file.getbuffer())
                content = extract_content_from_url(f.name)
                data = extract_structured_data(content, data_points)
                json_data = json.loads(data)

                if isinstance(json_data, list):
                    json_data[0]['invoice_name'] = file.name
                    results.extend(json_data)  # Use extend() for lists
                else:
                    json_data['invoice_name'] = file.name
                    results.append(json_data)  # Wrap the dict in a list

        if len(results) > 0:
            try:
                print(len(results))
                df = pd.DataFrame(results)
                st.subheader("Results")
                st.data_editor(df)

            except Exception as e:
                st.error(
                    f"An error occurred while creating the DataFrame: {e}")
                st.write(results)  # Print the data to see its content


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
