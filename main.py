import streamlit as st
import os
from openai import OpenAI
import tempfile
import time
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError
import logging
import hashlib
import pytesseract
from PIL import Image
from byaldi import RAGMultiModalModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Disable tokenizers parallelism to avoid deadlock issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_document_hash(file_path):
    # Generate a hash for the document to identify it uniquely
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def call_gpt4(api_key, content):
    """Calls the OpenAI GPT-4 API to generate structured instructions."""
    client = OpenAI(api_key=api_key)
    messages = [
        {
            "role": "user",
            "content": f"""
            The following is extracted from a gateway manual. Please provide step-by-step instructions based on this text. Respond with concise and clear procedural steps:
            {content}
            """
        }
    ]
    
    # Print the message dictionary being sent to GPT-4
    logger.info(f"Message context sent to GPT-4: {messages}")
    st.write("### GPT-4 Context Sent:")
    st.json(messages)  # Display the context as JSON in the Streamlit app

    # Call the GPT-4 API
    response = client.chat.completions.create(
        messages=messages,
        model="gpt-4",
    )

    # Print the entire GPT-4 response for inspection
    logger.info(f"GPT-4 API Response: {response}")
    st.write("### GPT-4 Raw Response:")
    st.json(response)  # Display the raw response as JSON in the Streamlit app

    # Access the correct content in the response
    try:
        return response.choices[0].message.content.strip()  # Corrected access to message content
    except Exception as e:
        logger.error(f"Error parsing GPT-4 response: {e}")
        st.error(f"Error parsing GPT-4 response: {e}")
        return "An error occurred."

def main():
    # Streamlit app title
    st.title("Document Query App with Colpali, Tesseract OCR, and GPT-4")

    # Get OpenAI API key from the user
    openai_api_key = st.text_input("Enter your OpenAI API key:", type="password")

    if not openai_api_key:
        st.warning("Please provide your OpenAI API key to proceed.")
        return

    # Load the Colpali model once and store in session state
    if 'model' not in st.session_state:
        with st.spinner("Initializing the Colpali model..."):
            logger.info("Initializing the Colpali model...")
            time.sleep(1)  # Simulate some loading time
            st.session_state.model = RAGMultiModalModel.from_pretrained("vidore/colpali")
        st.success("Colpali initialized successfully!")
        logger.info("Colpali initialized successfully!")

    # Assign the model from session state
    model = st.session_state.model

    # Create a persistent temporary directory for the session
    if 'temp_dir' not in st.session_state:
        st.session_state.temp_dir = tempfile.mkdtemp()
        st.session_state.file_path = None

    # Uploading the document if not already indexed
    if 'uploaded_file' not in st.session_state and ('indexed' not in st.session_state or not st.session_state.indexed):
        uploaded_file = st.file_uploader("Upload a PDF or DOC file", type=["pdf", "doc", "docx"])
        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
            st.session_state.indexed = False  # Ensure indexed state is reset for new uploads

            # Save the uploaded file to the persistent temporary directory
            file_path = os.path.join(st.session_state.temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.file_path = file_path
            logger.info(f"File saved to temporary directory: {file_path}")

            # Generate document hash to identify it uniquely
            document_hash = get_document_hash(file_path)
            st.session_state.document_hash = document_hash

    # Proceed with indexing if a file has been uploaded but not indexed yet
    if 'uploaded_file' in st.session_state and not st.session_state.indexed:
        file_path = st.session_state.file_path
        if file_path and os.path.exists(file_path):
            document_hash = st.session_state.document_hash
            index_dir = "persistent_indexes"
            os.makedirs(index_dir, exist_ok=True)
            index_path = os.path.join(index_dir, f"index_{document_hash}")

            # Check if index already exists
            if os.path.exists(index_path):
                logger.info("Loading existing index...")
                st.session_state.indexed = True
                st.success("Document index loaded successfully! You can now enter your query below.")
            else:
                # Step-wise progress for indexing the document
                try:
                    # Ensure the uploaded file is a supported type for indexing
                    if not file_path.endswith(('.pdf', '.doc', '.docx')):
                        raise ValueError("Unsupported input type: Only PDF and DOC/DOCX files are supported for indexing.")
                    
                    with st.spinner("Indexing the document, please wait..."):
                        logger.info("Indexing the document...")
                        time.sleep(2)  # Simulate indexing time
                        model.index(
                            input_path=file_path,  # Index the specific file directly
                            index_name=index_path,
                            store_collection_with_index=False,
                            overwrite=True
                        )
                    st.success("Document indexed successfully! You can now enter your query below.")
                    logger.info("Document indexed successfully!")
                    st.session_state.indexed = True
                
                except PDFInfoNotInstalledError:
                    st.error("Poppler is not installed. Please install Poppler to continue.")
                    logger.error("Poppler is not installed. Unable to proceed.")
                    st.session_state.indexed = False
                    return
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
                    logger.error(f"An unexpected error occurred: {e}")
                    st.session_state.indexed = False
                    return
        else:
            st.error("The uploaded file is no longer available. Please upload it again.")
            st.session_state.indexed = False
            return

    # Query input from the user after document indexing
    if st.session_state.get('indexed', False):
        st.write("---")
        st.subheader("Search the Document")
        query = st.text_input("Enter your search query here:")

        if st.button("Search") and query:
            with st.spinner("Performing search, please wait..."):
                logger.info(f"Performing search with query: {query}")
                time.sleep(1)  # Simulate searching time
                results = model.search(query, k=5)

            # Display the search results and extract the relevant pages
            if results:
                st.write("### Search Results:")
                logger.info("Search results found:")
                
                full_extracted_text = ""  # To store combined text for GPT-4

                for result in results:
                    try:
                        page_num = result.page_num
                    except AttributeError:
                        st.error("Result object does not have 'page_num' attribute.")
                        logger.error("Result object does not have 'page_num' attribute.")
                        continue

                    st.write(f"Processing page {page_num}...")
                    logger.info(f"Processing page {page_num}...")

                    try:
                        # Convert the specific page to an image with lower resolution to reduce memory usage
                        pages = convert_from_path(st.session_state.file_path, dpi=50, first_page=page_num, last_page=page_num)
                        if pages:
                            for page in pages:
                                st.image(page, caption=f"Page {page_num}", use_column_width=True)

                                # Extract text from the image using Tesseract OCR
                                text = pytesseract.image_to_string(page)
                                full_extracted_text += f"Page {page_num}: {text}\n"

                    except PDFInfoNotInstalledError:
                        st.error("Poppler is not installed. Please install Poppler to continue.")
                        logger.error("Poppler is not installed. Unable to proceed.")
                    except Exception as e:
                        st.error(f"An error occurred while processing page {page_num}: {e}")
                        logger.error(f"An error occurred while processing page {page_num}: {e}")
                
                # Call GPT-4 with the combined text and display the output
                st.write("### GPT-4 Generated Instructions:")
                with st.spinner("Generating instructions with GPT-4..."):
                    gpt_response = call_gpt4(openai_api_key, full_extracted_text)
                    st.write(gpt_response)

if __name__ == "__main__":
    main()
