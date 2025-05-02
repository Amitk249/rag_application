from transformers import BlipProcessor, BlipForConditionalGeneration
import os
import json
from PIL import Image
import argparse
from tqdm import tqdm
import torch


def load_model():
    """
    load the Blip image captioning model and processor
    returnn processor and model objects

    """
    print("loading blip model")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Model loaded successfully. Using Device:{device}")

    return processor, model, device

def generate_caption(image_path , processor, model, device):
    """
    generate a caption for a single image using Blip

    Args:
        image_path = path to the image file
        proecessor = blip processor
        model = blip model
        device = device to run the model on cuda or cpu
    """
    try:
        image = Image.open(image_path).convert('RGB')
        inputs = processor(images=image, return_tensors='pt').to(device)

        # generate caption
        with torch.no_grad():
            out = model.generate(**inputs)
        
        # Decode the caption
        caption = processor.decode(out[0], skip_special_tokens = True)

        # Add fashion specific context if not present
        if not any(word in caption.lower() for word in ['clothing' , 'dress', 'pants', 'outfits', 'wear', 'fashion']):
            caption = f"A Fashion Item Showing{caption}"

        return caption
    except Exception as e:
        print(f"Error processing {image_path}:{e}")
        return None
    
def process_image_folder(image_folder, output_file, limit = None):
    """
    Process all the images in a folder, generate caption, and save results.

    args:
        image folder = path to the folder containing images
        output_file = path to save the resulting json/csv file
        limit = optional maximum number of images to process

    """
    # load the model
    processor, model , device = load_model()

    # Get list of image files
    valid_extentions = {'.jpg', '.jpeg','.png','.webp','.bmp'}
    image_files = [
        f for f in os.listdir(image_folder)
        if os.path.isfile(os.path.join(image_folder,f)) and os.path.splitext(f.lower()) [1] in valid_extentions

    ]

    if limit and limit > 0:
        image_files = image_files[:limit]
    print(f"Found{len(image_files)} images to process")

    # process each image
    results = []
    for image_name in tqdm(image_files, desc='Processing_images'):
        image_path = os.path.join(image_folder, image_name)
        caption = generate_caption(image_path, processor, model, device)

        if caption:
            results.append({
                'image_name':image_name,
                'caption':caption,
                'image_path':image_path
            })
    # Save results
    with open(output_file,'w') as f:
        json.dump(results, f, indent=2)

    print(f"processed{len(results)} images successfully")
    print(f"Results saved to{output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate captions for fashion images using BLIP')
    parser.add_argument('--image_folder', type=str, required=True, help='Path to folder containing images')
    parser.add_argument('--output_file', type=str, default='fashion_captions.json', help='Path to save output JSON file')
    parser.add_argument('--limit', type=int, default=None, help='Maximum number of images to process (for testing)')
    
    args = parser.parse_args()
    
    process_image_folder(args.image_folder, args.output_file, args.limit)

if __name__ == "__main__":
    main()