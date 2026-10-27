
from gliner import GLiNER

LABELS = ["PERSON", "ORG", "LOCATION", "EVENT"]


class GLiNERRunner:
    def __init__(self, threshold=0.4):
        print("Loading GLiNER model...")
        self.model = GLiNER.from_pretrained(
            "urchade/gliner_medium-v2.1"
        )
        self.threshold = threshold
        print("GLiNER model loaded successfully!")

    def predict(self, text):
        predictions = self.model.predict_entities(
            text,
            LABELS,
            threshold=self.threshold,
        )

        entities = []

        for item in predictions:
            entities.append({
                "text": item["text"],
                "label": item["label"],
            })

        return {"entities": entities}


if __name__ == "__main__":
    runner = GLiNERRunner()

    sample = (
        "Narendra Modi met Microsoft executives in New Delhi "
        "during the summit."
    )

    result = runner.predict(sample)
    print(result)
