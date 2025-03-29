import huggingface_hub

from PIL import Image

MODEL_REPO_PREFIX = "MiaoshouAI/" 

FLORENCE2_PROMPTGEN_LARGE = "MiaoshouAI/Florence-2-large-PromptGen-v2.0"
FLORENCE2_PROMPTGEN_BASE = "MiaoshouAI/Florence-2-base-PromptGen-v2.0"

class Predictor:
    def __init__(self):
        self.model = None
        self.processor = None
        self.prompt = "<GENERATE_TAGS>"

    def load_model(self, model_repo: str):
        from yadt.tagger_florence2_promptgen_model import load_model

        if model_repo == FLORENCE2_PROMPTGEN_LARGE:
            repo_name = "MiaoshouAI/Florence-2-large-PromptGen-v2.0"
            revision = "4aa33eaf50aab040fe8523312ff52eb53322c220"
        elif model_repo == FLORENCE2_PROMPTGEN_BASE:
            repo_name = "MiaoshouAI/Florence-2-base-PromptGen-v2.0"
            revision = "59b6e4bf75d0f3e8a6b1a14211f6a50fcdd48d63"
        else:
            raise AssertionError(f"Unsupported model repo: {model_repo}")
        
        self.model, self.processor = load_model(repo_name=repo_name, revision=revision)


    def predict(self, image: Image):
        assert self.model is not None, "No model loaded"
        assert self.processor is not None, "No model processor loaded"

        if getattr(image, "mode", "NOT_RGB") != "RGB":
            rgb_image = Image.new("RGB", image.size, (255, 255, 255))
            rgb_image.paste(image)
            image = rgb_image

        inputs = self.processor(text=self.prompt, images=image, return_tensors="pt").to('cpu')

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        parsed_answer: str = self.processor.post_process_generation(generated_text, task=self.prompt, image_size=(image.width, image.height))

        return {}, { tag.strip(): 1.0 for tag in parsed_answer[self.prompt].split(',')}, {}
