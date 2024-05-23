"""
01_save_model.py

Script to save wd14 model to model store of bentoml
for further container build
"""

import bentoml
from huggingface_hub import from_pretrained_keras

def load_keras_tagger_hf(tagger):
    """Loads tagger from hugging_face"""
    return from_pretrained_keras(tagger, compile=False)


if __name__ == "__main__":
    TAGGER = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
    model = load_keras_tagger_hf(TAGGER)

    # Write some model info for verification
    input_shape = model.input_shape[1:3]
    output_shape = model.output_shape

    print("===== MODEL INFO =====")
    print("=> Model summary")
    model.summary(print_fn=lambda x: print(f"=> {x}"), line_length=120)
    print(f"=> Input shape:\t{input_shape}")
    print(f"=> Output shape:\t{output_shape}")
    print("======================")

    print("Saving model to bentoml model store...")
    bentoml_model = bentoml.keras.save_model("wd14-remy", model)
    print(f"Model saved: {bentoml_model}")
