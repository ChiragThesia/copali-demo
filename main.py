import streamlit as st
import os
import tempfile
import time
import shutil
from io import BytesIO
import requests
from PyPDF2 import PdfReader, PdfWriter
from pdf2image import convert_from_path
from byaldi import RAGMultiModalModel
import hashlib
import base64

# Disable parallelism to avoid issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Hardcoded API key
API_KEY = "Gw_KoFGGix1gF9KzSg9getLt4ML-343asZkuAWAnv2Q"  # Replace with your actual API key

# Helper to compute a hash for the document
def get_document_hash(file_path):
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

# Function to initialize the Colpali model
def initialize_colpali_model():
    if 'model' not in st.session_state:
        with st.spinner("Initializing the Colpali model..."):
            time.sleep(1)  # Simulate loading time
            st.session_state.model = RAGMultiModalModel.from_pretrained("vidore/colpali")
        st.success("Colpali initialized successfully!")
    return st.session_state.model

# Function to save the uploaded file
def save_uploaded_file(uploaded_file):
    if 'temp_dir' in st.session_state:
        try:
            shutil.rmtree(st.session_state.temp_dir, ignore_errors=True)
        except Exception as e:
            st.error(f"Error cleaning temp directory: {e}")
    
    st.session_state.temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(st.session_state.temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.session_state.file_path = file_path
    st.session_state.document_hash = get_document_hash(file_path)
    return file_path

# Function to index the uploaded PDF or DOC file using Colpali
def index_document(file_path, model):
    document_hash = st.session_state.document_hash
    index_dir = "persistent_indexes"
    os.makedirs(index_dir, exist_ok=True)
    index_path = os.path.join(index_dir, f"index_{document_hash}")

    with st.spinner("Indexing the document, please wait..."):
        try:
            model.index(
                input_path=file_path,
                index_name=index_path,
                store_collection_with_index=False,
                overwrite=True
            )
            st.success("Document indexed successfully!")
            st.session_state.indexed = True
        except Exception as e:
            st.error(f"Error during indexing: {e}")
            st.session_state.indexed = False

# Function to perform search on the indexed document
def perform_search(query, model):
    if st.session_state.indexed:
        with st.spinner("Performing search, please wait..."):
            try:
                return model.search(query, k=5)
            except Exception as e:
                st.error(f"Error during search: {e}")
                return []
    else:
        st.error("Document not indexed. Please upload and index a document first.")
        return []

# Function to create a new PDF with specific pages based on the search results
def create_pdf_with_pages(file_path, page_numbers):
    reader = PdfReader(file_path)
    writer = PdfWriter()

    # Add the specified pages to the new PDF
    for page_num in page_numbers:
        writer.add_page(reader.pages[page_num - 1])  # Page numbers are 1-based

    # Save the new PDF to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        writer.write(temp_pdf)
        temp_pdf_path = temp_pdf.name

    return temp_pdf_path

# Function to send the newly created PDF to the /api/v1/marker endpoint and poll for results
def send_pdf_to_marker_api(pdf_path, langs, max_pages, force_ocr, paginate, extract_images):
    url = "https://www.datalab.to/api/v1/marker"
    headers = {"X-API-Key": API_KEY}
    # Using 'with' to manage the file context safely
    with open(pdf_path, "rb") as file_data:
        files = {"file": ('extracted.pdf', file_data, 'application/pdf')}
        data = {
            "langs": langs,
            "max_pages": str(max_pages),
            "force_ocr": str(force_ocr).lower(),
            "paginate": str(paginate).lower(),
            "extract_images": str(extract_images).lower()
        }

        try:
            with st.spinner("Sending PDF to Marker API..."):
                # Send the initial request
                response = requests.post(url, headers=headers, files=files, data=data)
                response.raise_for_status()
                response_data = response.json()

                # Get the request_check_url from the response
                check_url = response_data["request_check_url"]
                st.write(f"Polling URL: {check_url}")

                # Poll the request_check_url until we get the status "complete"
                return poll_for_result(check_url, headers)

        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {e}")
            return None

# Function to poll the request_check_url for the processing result
def poll_for_result(check_url, headers):
    poll_interval = 2  # Start polling every 2 seconds
    max_polls = 300  # Max number of polls to avoid infinite loop

    for i in range(max_polls):
        time.sleep(poll_interval)

        # Perform the request to check the status
        try:
            response = requests.get(check_url, headers=headers)
            response.raise_for_status()
            data = response.json()  # Parse JSON response here

            # Check the status
            if data.get("status") == "complete":
                return data
            elif data.get("status") == "failed":
                st.error(f"Processing failed: {data.get('error')}")
                return None

            # Exponential backoff
            poll_interval = min(poll_interval * 2, 60)  # Backoff to a max of 60 seconds

        except requests.exceptions.RequestException as e:
            st.error(f"Polling request failed: {e}")
            return None

    st.error("Polling timed out.")
    return None


# Main function to manage the Streamlit app
def main():
    st.set_page_config(layout="wide")
    
    # Initialize session state for indexing
    if 'indexed' not in st.session_state:
        st.session_state.indexed = False

    # Initialize results variable
    results = []

    # Initialize the Colpali model
    model = initialize_colpali_model()

    # Sidebar: File upload and indexing
    st.sidebar.title("Colpali PDF Indexing and Querying")
    uploaded_file = st.sidebar.file_uploader("Upload a PDF or DOC file", type=["pdf", "doc", "docx"])
    
    if uploaded_file:
        if not st.session_state.indexed:
            file_path = save_uploaded_file(uploaded_file)
            index_document(file_path, model)
        else:
            st.sidebar.write("Document is already indexed.")

        # Allow query input once the document is indexed
        if st.session_state.indexed:
            query = st.sidebar.text_input("Enter your search query here:")
            if st.sidebar.button("Search") and query:
                results = perform_search(query, model)

                # Display page numbers from the search results
                if results:
                    st.sidebar.write("### Search Results (Page Numbers):")
                    page_numbers = [result.page_num for result in results]
                    st.sidebar.write(page_numbers)

                    # Create a new PDF with only the pages from the search results
                    new_pdf_path = create_pdf_with_pages(st.session_state.file_path, page_numbers)

                    # Store the new PDF in a separate state for later use
                    st.session_state.new_search_pdf_path = new_pdf_path

                    # Provide a download link for the new PDF
                    with open(st.session_state.new_search_pdf_path, "rb") as f:
                        new_pdf_data = f.read()
                    st.sidebar.download_button(label="Download Extracted PDF", data=new_pdf_data, file_name="extracted_search_results.pdf")

                    max_pages = st.sidebar.number_input("Max Pages", min_value=1, value=5)
                    langs = st.sidebar.text_input("Languages", value="en")
                    force_ocr = st.sidebar.checkbox("Force OCR", value=False)
                    paginate = st.sidebar.checkbox("Paginate", value=False)
                    extract_images = st.sidebar.checkbox("Extract Images", value=True)

                    # Send the PDF to the API
                    api_response = send_pdf_to_marker_api(
                        pdf_path=new_pdf_path,
                        max_pages=max_pages,
                        langs=langs,
                        force_ocr=force_ocr,
                        paginate=paginate,
                        extract_images=extract_images
                    )

                    # Display API response
                    if api_response:
                        st.sidebar.write("### API Response")
                        st.sidebar.json(api_response)

                        # Save relevant data
                        if api_response.get("status") == "complete":
                            st.write("### Markdown")
                            st.write(api_response.get("markdown"))

                            st.write("### Metadata")
                            st.json(api_response.get("meta"))

                            st.write("### Extracted Images")
                            for img_name, img_data in api_response.get("images", {}).items():
                                # Decode base64 string to bytes if necessary
                                if isinstance(img_data, str):
                                    img_data = base64.b64decode(img_data)
                                st.image(BytesIO(img_data), caption=img_name)

    # Main page: Instructions or other content
    st.title("Welcome to the Colpali PDF Tool")

    # Right column: Preview the newly created PDF from search results
    if 'new_search_pdf_path' in st.session_state:
        st.write("### PDF from Search Results")
        try:
            # Convert the entire PDF to images
            images = convert_from_path(st.session_state.new_search_pdf_path, dpi=150)
            
            # Display images in a grid with 3 columns
            for i in range(0, len(images), 3):
                cols = st.columns(3)
                for j, img in enumerate(images[i:i + 3]):
                    cols[j].image(img, use_column_width=True)
        except Exception as e:
            st.error(f"Could not preview the new PDF: {e}")

if __name__ == "__main__":
    main()
