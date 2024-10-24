import streamlit as st
import os
from openai import OpenAI
import tempfile
import time
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError
import logging
import logging.config
import hashlib
from PIL import Image
from huggingface_hub import InferenceClient
from io import BytesIO
from byaldi import RAGMultiModalModel
import shutil
from huggingface_hub import InferenceClient
from io import BytesIO

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Disable tokenizers parallelism to avoid deadlock issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Hugging Face API URL
API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-11B-Vision-Instruct"

def configure_logging():
    """Configure logging settings"""
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
        },
        'handlers': {
            'default': {
                'level': 'INFO',
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
            },
        },
        'loggers': {
            '': {
                'handlers': ['default'],
                'level': 'INFO',
                'propagate': True
            }
        }
    }
    logging.config.dictConfig(logging_config)
    return logging.getLogger(__name__)

def get_document_hash(file_path):
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def initialize_colpali_model():
    if 'model' not in st.session_state:
        with st.spinner("Initializing the Colpali model..."):
            logger.info("Initializing the Colpali model...")
            time.sleep(1)  # Simulate loading time
            st.session_state.model = RAGMultiModalModel.from_pretrained("vidore/colpali")
        st.success("Colpali initialized successfully!")
        logger.info("Colpali initialized successfully!")
    return st.session_state.model

def save_uploaded_file(uploaded_file):
    # Clear any existing temp directories and files
    if 'temp_dir' in st.session_state:
        try:
            shutil.rmtree(st.session_state.temp_dir, ignore_errors=True)
        except Exception as e:
            logger.error(f"Error cleaning temp directory: {e}")
    
    # Create new temp directory
    st.session_state.temp_dir = tempfile.mkdtemp()
    
    # Save the new file
    file_path = os.path.join(st.session_state.temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.session_state.file_path = file_path
    st.session_state.document_hash = get_document_hash(file_path)
    logger.info(f"File saved to temporary directory: {file_path}")
    return file_path

def clean_index_directory(index_path):
    """Clean up existing index directory and .byaldi directory"""
    # Clean persistent_indexes directory
    if os.path.exists(index_path):
        try:
            shutil.rmtree(index_path)
            logger.info(f"Cleaned up existing index at: {index_path}")
        except Exception as e:
            logger.error(f"Error cleaning up index directory: {e}")
    
    # Clean .byaldi directory
    byaldi_dir = os.path.join(os.getcwd(), '.byaldi')
    if os.path.exists(byaldi_dir):
        try:
            shutil.rmtree(byaldi_dir)
            logger.info(f"Cleaned up .byaldi directory at: {byaldi_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up .byaldi directory: {e}")

def index_document(file_path, model):
    if not file_path.endswith(('.pdf', '.doc', '.docx')):
        raise ValueError("Unsupported input type: Only PDF and DOC/DOCX files are supported for indexing.")
    
    document_hash = st.session_state.document_hash
    index_dir = "persistent_indexes"
    os.makedirs(index_dir, exist_ok=True)
    index_path = os.path.join(index_dir, f"index_{document_hash}")

    # Clean up existing indexes
    clean_index_directory(index_path)

    with st.spinner("Indexing the document, please wait..."):
        logger.info("Indexing the document...")
        try:
            model.index(
                input_path=file_path,
                index_name=index_path,
                store_collection_with_index=False,
                overwrite=True
            )
            st.success("Document indexed successfully!")
            logger.info("Document indexed successfully!")
            st.session_state.indexed = True
        except Exception as e:
            logger.error(f"Error during indexing: {e}")
            st.error(f"Error during indexing: {e}")
            st.session_state.indexed = False

def call_gpt4o(api_key, content, image_descriptions_dict):
    client = OpenAI(api_key=api_key)
    
    formatted_descriptions = "\n\n".join([
        f"Page {page_num}:\n"
        f"Location: {info['location']}\n"
        f"Description: {info['description']}"
        for page_num, info in image_descriptions_dict.items()
    ])
    
    enhanced_content = f"""
    User Query: {content}
    
    Available Technical Diagrams and Images:
    {formatted_descriptions}
    
    Based on the above query and the technical diagrams found in the documentation, 
    please provide detailed step-by-step installation instructions. Reference specific 
    diagrams by their page numbers where relevant.
    """
    
    messages = [{"role": "user", "content": enhanced_content}]
    logger.info("Sending structured content to GPT-4...")
    st.write("### Enhanced Content Sent to GPT-4:")
    st.json(image_descriptions_dict)
    
    response = client.chat.completions.create(
        messages=messages,
        model="gpt-4o"
    )
    
    try:
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error parsing GPT-4 response: {e}")
        st.error(f"Error parsing GPT-4 response: {e}")
        return "An error occurred."

def perform_search(query, model):
    if st.session_state.indexed:
        with st.spinner("Performing search, please wait..."):
            logger.info(f"Performing search with query: {query}")
            try:
                return model.search(query, k=5)
            except Exception as e:
                logger.error(f"Error during search: {e}")
                st.error(f"Error during search: {e}")
                return []
    else:
        st.error("Document not indexed. Please upload and index a document first.")
        return []

def extract_images_from_pdf(pdf_path, page_numbers):
    """Extract images from specific PDF pages with location metadata."""
    images_dict = {}
    
    for page_num in page_numbers:
        try:
            pages = convert_from_path(pdf_path, dpi=200, first_page=page_num, last_page=page_num)
            if pages:
                for page in pages:
                    # Store both the image and its location metadata
                    images_dict[page_num] = {
                        'image': page,
                        'location': f'Page {page_num}',
                        'description': None  # Will be filled by Hugging Face Vision
                    }
                    
        except Exception as e:
            logger.error(f"Error processing page {page_num}: {e}")
            continue
    
    return images_dict

def call_huggingface_vision(api_key, images_dict):
    """Process images using Hugging Face's InferenceClient for captioning."""
    client = InferenceClient(api_key=api_key)
    
    for page_num, info in images_dict.items():
        image = info['image']

        logger.info(f"Processing image from page {page_num} with Hugging Face Vision...")
        
        try:
            # Convert the PIL image to JPEG format
            buffered = BytesIO()
            image.convert("RGB").save(buffered, format="JPEG")
            buffered.seek(0)
            
            # Use the client.image_to_text method to generate image captions
            caption = client.image_to_text(buffered)
            images_dict[page_num]['description'] = caption
        
        except Exception as e:
            logger.error(f"Error processing image on page {page_num}: {e}")
            images_dict[page_num]['description'] = f"Error processing image: {str(e)}"
            continue

    return images_dict

def main():
    st.title("Colpali demo SureSteps")

    # Initialize session state
    if 'indexed' not in st.session_state:
        st.session_state.indexed = False

    # API key inputs
    openai_api_key = st.text_input("Enter your OpenAI API key:", type="password")
    huggingface_api_key = st.text_input("Enter your Hugging Face API key:", type="password")
    
    if not openai_api_key or not huggingface_api_key:
        st.warning("Please provide both OpenAI and Hugging Face API keys to proceed.")
        return

    logger = configure_logging()
    
    # Initialize Colpali once at the start
    model = initialize_colpali_model()
    
    # Clean up directories
    index_dir = "persistent_indexes"
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)
    os.makedirs(index_dir, exist_ok=True)
    
    byaldi_dir = os.path.join(os.getcwd(), '.byaldi')
    if os.path.exists(byaldi_dir):
        shutil.rmtree(byaldi_dir)

    uploaded_file = st.file_uploader("Upload a PDF or DOC file", type=["pdf", "doc", "docx"])
    if uploaded_file:
        if not st.session_state.indexed:
            file_path = save_uploaded_file(uploaded_file)
            index_document(file_path, model)
        else:
            st.write("Document is already indexed.")

    if st.session_state.indexed:
        query = st.text_input("Enter your search query here:")
        if st.button("Search") and query:
            # Step 1: Get relevant pages from Colpali
            results = perform_search(query, model)
            
            if results:
                # Step 2: Extract page numbers
                page_numbers = [result.page_num for result in results if hasattr(result, 'page_num')]
                
                # Step 3: Extract images from those pages with metadata
                st.write("### Extracting Images from PDF...")
                images_dict = extract_images_from_pdf(st.session_state.file_path, page_numbers)
                
                if images_dict:
                    # Display original images
                    st.write("### Original Images:")
                    for page_num, info in images_dict.items():
                        st.image(info['image'], caption=f"Page {page_num}", use_column_width=True)
                    
                    # Step 4: Get image descriptions from Hugging Face
                    st.write("### Generating Image Descriptions...")
                    images_dict = call_huggingface_vision(huggingface_api_key, images_dict)
                    
                    # Step 5: Generate installation instructions with GPT-4o
                    st.write("### Generating Installation Instructions...")
                    installation_instructions = call_gpt4o(openai_api_key, query, images_dict)
                    
                    st.write("### Final Installation Instructions:")
                    st.write(installation_instructions)

if __name__ == "__main__":
    main()