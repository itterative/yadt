def load_model(device='cpu', **kwargs):
    from transformers import AutoModelForCausalLM, AutoProcessor

    repo_name = kwargs.pop('repo_name', 'MiaoshouAI/Florence-2-large-PromptGen-v2.0')
    revision = kwargs.pop('revision', "4aa33eaf50aab040fe8523312ff52eb53322c220")

    model = AutoModelForCausalLM.from_pretrained(repo_name, revision=revision, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(repo_name, revision=revision, trust_remote_code=True)

    return model, processor


# Example usage
if __name__ == "__main__":
    import sys
    from PIL import Image
    
    # Get model directory from command line or use default
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "./exported_model"
    

    # Load model
    model, processor = load_model(model_dir)
    
    # Display info
    print(f"\nModel information:")
    print(f"  Pretrained: MiaoshouAI/Florence-2-large-PromptGen-v2.0")
    

    prompt = "<GENERATE_TAGS>"
    
    # Test on an image if provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"\nRunning inference on {image_path}")
        
        image = Image.open(image_path)
        image = image.convert("RGB")

        inputs = processor(text=prompt, images=image, return_tensors="pt").to('cpu')

        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        parsed_answer = processor.post_process_generation(generated_text, task=prompt, image_size=(image.width, image.height))

        print(parsed_answer)
        print("")

