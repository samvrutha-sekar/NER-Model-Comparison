


import json
import re
import time
import ollama


LABELS = ["PERSON", "ORG", "LOCATION", "EVENT"]

SYSTEM_PROMPT = """
You are a named entity recognition system.

Extract only entities explicitly present in the input text.
Valid labels: PERSON, ORG, LOCATION, EVENT.

Return one JSON object:
{"entities":[{"text":"Priya Nair","label":"PERSON"}]}

Rules:
- PERSON: names of people.
- ORG: companies and organizations.
- LOCATION: geographical places.
- EVENT: named events.
- Do not invent or duplicate entities.
- Use exact text spans from the input.
- Use the key "label".
- Return {"entities":[]} when no entities are found.
- Do not include explanations or markdown.
"""


class LLMRunner:
    def __init__(self, model_name="qwen3:4b", timeout_seconds=90):
        self.model_name = model_name
        self.client = ollama.Client(timeout=timeout_seconds)

        print(f"Ollama model: {self.model_name}")
        print(f"Request timeout: {timeout_seconds} seconds")

    def predict(self, text):
        response = self.client.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": (
                        "Extract named entities from this text:\n"
                        + text
                    ),
                },
            ],
            # Ask Qwen3 not to generate a thinking trace.
            think=False,
            format={
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "label": {
                                    "type": "string",
                                    "enum": LABELS,
                                },
                            },
                            "required": ["text", "label"],
                        },
                        "additionalProperties": False,
                    }
                },
                "required": ["entities"],
                "additionalProperties": False,
            },
            options={
                "temperature": 0,
                "num_predict": 256,
                "num_ctx": 2048,
            },
        )

        raw = response["message"]["content"]

        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            # Defensive fallback if the model returns extra text.
            cleaned = re.sub(
                r"```(?:json)?|```",
                "",
                raw,
                flags=re.IGNORECASE,
            ).strip()

            match = re.search(r"\{.*\}", cleaned, re.DOTALL)

            if not match:
                print("JSON parsing failed; returning no entities.")
                return {"entities": []}

            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                print("Malformed JSON; returning no entities.")
                return {"entities": []}

        if not isinstance(data, dict):
            return {"entities": []}

        entities = data.get("entities", [])

        if not isinstance(entities, list):
            return {"entities": []}

        valid = []

        for entity in entities:
            if not isinstance(entity, dict):
                continue

            entity_text = entity.get("text")
            label = entity.get("label")

            if (
                isinstance(entity_text, str)
                and entity_text.strip()
                and isinstance(label, str)
                and label in LABELS
            ):
                valid.append({
                    "text": entity_text.strip(),
                    "label": label,
                })

        return {"entities": valid}


if __name__ == "__main__":
    runner = LLMRunner(model_name="qwen3:4b")

    sample = (
        "Priya Nair joined Microsoft in Singapore "
        "during the Lantern Festival."
    )

    start = time.perf_counter()

    try:
        result = runner.predict(sample)

        print("\nPrediction:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(
            f"\nElapsed time: "
            f"{time.perf_counter() - start:.2f} seconds"
        )

    except Exception as exc:
        print(f"\nOllama request failed: {exc}")
