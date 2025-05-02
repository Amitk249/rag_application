"""
Fashion Image Search - Streamlit Web UI

This application provides a web interface for searching fashion images using:
1. Text queries (semantic search)
2. Image uploads (visual similarity search)

Usage:
    streamlit run app.py
"""

import os
import streamlit as st
from PIL import Image
import tempfile
import json
import numpy as np
from src.search_engine import FashionSearchEngine

# Page configuration
st.set_page_config(
    page_title="Fashion Image Search",
    page_icon="👗",
    layout="wide",
)

# Constants
DATA_DIR = "data"
INDEX_PATH = os.path.join(DATA_DIR, "captions.index")
IMAGE_PATHS_FILE = os.path.join(DATA_DIR, "image_paths.json")
CAPTIONS_FILE = os.path.join(DATA_DIR, "processed_captions.json")

# Cache the search engine initialization
@st.cache_resource
def load_search_engine():
    """Load the search engine (cached to avoid reloading)"""
    if not os.path.exists(INDEX_PATH) or not os.path.exists(IMAGE_PATHS_FILE):
        st.error("Index files not found. Please run the index builder first.")
        st.stop()
    
    return FashionSearchEngine(INDEX_PATH, IMAGE_PATHS_FILE)

def load_captions():
    """Load captions data if available"""
    if not os.path.exists(CAPTIONS_FILE):
        return {}
    
    with open(CAPTIONS_FILE, 'r') as f:
        captions_data = json.load(f)
    
    return {item['image_path']: item['caption'] for item in captions_data}

def display_results(results, captions_dict):
    """Display search results in a grid layout"""
    if not results:
        st.warning("No results found.")
        return
    
    # Create columns for the results
    cols = st.columns(3)
    
    for i, result in enumerate(results):
        col_idx = i % 3
        with cols[col_idx]:
            image_path = result['image_path']
            similarity = result['similarity']
            
            # Get caption if available
            caption = captions_dict.get(image_path, "No caption available")
            
            # Display the image
            try:
                img = Image.open(image_path)
                st.image(img, caption=f"Match #{i+1}", use_column_width=True)
                
                # Display metadata
                st.write(f"**Similarity:** {similarity:.2f}")
                st.write(f"**Caption:** {caption}")
                
                # Add a separator
                st.markdown("---")
            except Exception as e:
                st.error(f"Error displaying image {image_path}: {e}")

def generate_description(results, query):
    """Generate a description of the search results"""
    if not results:
        return "I couldn't find any matching fashion items. Try a different search query."
    
    # Count item types
    item_types = {}
    for result in results:
        if 'caption' in result:
            caption = result['caption'].lower()
            # Extract key fashion terms
            for term in ['dress', 'shirt', 'pants', 'skirt', 'jacket', 'suit', 'saree', 
                        'top', 'jeans', 'coat', 'blouse', 'sweater', 'tshirt', 't-shirt']:
                if term in caption:
                    item_types[term] = item_types.get(term, 0) + 1
    
    # Create description
    description = f"Here are {len(results)} fashion items matching '{query}'.\n\n"
    
    if item_types:
        description += "I found: "
        items_list = [f"{count} {item_type}{'s' if count > 1 else ''}" 
                    for item_type, count in item_types.items()]
        description += ", ".join(items_list) + ".\n\n"
    
    description += "The matches are sorted by similarity to your query. Hope you find something you like!"
    
    return description

def main():
    # Header
    st.title("👗 Fashion Image Search")
    st.write("Search for fashion items using text or image uploads")
    
    # Sidebar setup
    st.sidebar.title("Search Options")
    search_type = st.sidebar.radio("Search Type", ["Text Search", "Image Search"])
    top_k = st.sidebar.slider("Number of results", min_value=3, max_value=20, value=9, step=3)
    
    # Load the search engine
    search_engine = load_search_engine()
    
    # Load captions
    captions_dict = load_captions()
    
    # Main content area
    if search_type == "Text Search":
        st.subheader("Find Fashion Items by Description")
        query = st.text_input("Describe what you're looking for:", placeholder="e.g., red floral summer dress")
        
        if st.button("Search") or query:
            if not query:
                st.warning("Please enter a search query.")
            else:
                with st.spinner("Searching..."):
                    # Perform search
                    results = search_engine.search_by_text(query, top_k=top_k)
                    
                    # Display conversational description (optional Claude integration)
                    with st.expander("Search Summary", expanded=True):
                        st.markdown(generate_description(results, query))
                    
                    # Display search results
                    display_results(results, captions_dict)
    
    else:  # Image Search
        st.subheader("Find Similar Fashion Items")
        uploaded_file = st.file_uploader("Upload an image of a fashion item:", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display uploaded image
            col1, col2 = st.columns([1, 2])
            with col1:
                img = Image.open(uploaded_file)
                st.image(img, caption="Uploaded Image", use_column_width=True)
            
            # Search button
            with col2:
                st.write("Looking for fashion items similar to your uploaded image")
                search_button = st.button("Find Similar Items")
            
            if search_button:
                with st.spinner("Searching for similar items..."):
                    # Save uploaded image to a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                        img.save(tmp_file.name)
                        tmp_file_path = tmp_file.name
                    
                    try:
                        # Perform image search
                        results = search_engine.search_by_image(tmp_file_path, top_k=top_k)
                        
                        # Add captions to results
                        for result in results:
                            image_path = result['image_path']
                            if image_path in captions_dict:
                                result['caption'] = captions_dict[image_path]
                        
                        # Display search results
                        st.subheader("Similar Fashion Items")
                        display_results(results, captions_dict)
                    
                    except Exception as e:
                        st.error(f"Error performing image search: {e}")
                    
                    finally:
                        # Clean up temporary file
                        if os.path.exists(tmp_file_path):
                            os.unlink(tmp_file_path)

    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using FAISS, SentenceTransformers, and Streamlit")

if __name__ == "__main__":
    main()