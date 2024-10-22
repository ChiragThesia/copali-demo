import streamlit as st
from byaldi import RAGMultiModalModel
import os
import tempfile
import time
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError
import logging
import hashlib
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import torch
from PIL import Image

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

def main():
    # Streamlit app title
    st.title("Document Query App with Qwen2-VL Integration")
    st.write("Upload a document and query it using Qwen2-VL's multimodal capabilities.")

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

    # Load Qwen2-VL model for multimodal understanding if not already loaded
    if 'qwen2vl_model' not in st.session_state:
        with st.spinner("Loading Qwen2-VL model for image and text understanding..."):
            st.session_state.qwen2vl_model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.float16)
            st.session_state.qwen2vl_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        st.success("Qwen2-VL model loaded successfully!")
        logger.info("Qwen2-VL model loaded successfully!")

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
                images_to_process = []

                for result in results:
                    st.write(f"**Result Object**: {result}")
                    logger.info(f"Full Result: {result}")
                    # Extract and display the page from the PDF
                    if hasattr(result, 'page_num'):
                        page_num = result.page_num
                        try:
                            # Convert the specific page to an image with lower resolution to reduce memory usage
                            pages = convert_from_path(st.session_state.file_path, dpi=50, first_page=page_num, last_page=page_num)
                            if pages:
                                for page in pages:
                                    # Resize the image to further reduce memory consumption
                                    resized_image = page.resize((400, 300))
                                    st.image(resized_image, caption=f"Page {page_num}", use_column_width=True)
                                    images_to_process.append(resized_image)
                        except PDFInfoNotInstalledError:
                            st.error("Poppler is not installed. Please install Poppler to continue.")
                            logger.error("Poppler is not installed. Unable to proceed.")
                        except Exception as e:
                            st.error(f"An error occurred while extracting page {page_num}: {e}")
                            logger.error(f"An error occurred while extracting page {page_num}: {e}")

                # Use Qwen2-VL to generate insights from the collected images
                if images_to_process:
                    st.write("---")
                    st.subheader("Generated Response")
                    extracted_text = ""
                    logger.info("Processing images in bulk through Qwen2-VL...")
                    
                    for image in images_to_process:
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": image},
                                    {"type": "text", "text": query}
                                ]
                            }
                        ]
                        # Prepare the input for Qwen2-VL
                        text = st.session_state.qwen2vl_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        inputs = st.session_state.qwen2vl_processor(
                            text=[text],
                            images=[image],
                            padding=True,
                            return_tensors="pt"
                        )

                        # Ensure inputs are on CPU since we're using Streamlit's environment
                        inputs = inputs.to('cpu')

                        # Generate response using Qwen2-VL with reduced max tokens
                        logger.info("Generating response using Qwen2-VL...")
                        st.write("Generating response using Qwen2-VL...")

                        generated_ids = st.session_state.qwen2vl_model.generate(**inputs, max_new_tokens=64)  # Reduce tokens
                        generated_ids_trimmed = [
                            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                        ]
                        response = st.session_state.qwen2vl_processor.batch_decode(
                            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )[0]
                        extracted_text += response + "\n"

                    # Display the generated structured response
                    st.write(extracted_text)
            else:
                st.write("No results found for your query.")
                logger.info("No results found for the query.")

if __name__ == "__main__":
    main()
