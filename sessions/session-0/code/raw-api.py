from openai import OpenAI
from utils import load_gsm8k, evaluate

def raw_api_predict(question):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": "You are a math problem solver. Answer with just the final number."},
            {"role": "user", "content": question}
        ],
        max_tokens=50
    )
    # Simple class to mimic DSPy Prediction structure
    class Prediction:
        def __init__(self, answer):
            self.answer = answer
    return Prediction(response.choices[0].message.content.strip())

def main():
    testset = load_gsm8k()
    accuracy = evaluate(raw_api_predict, testset)
    print(f"Raw API Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
