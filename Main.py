# Imports
import warnings
from transformers import pipeline, logging

# Ignore warnings and set logging level to error
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# Test Variables for u
test_para_1 = """Name: John Doe
Age: 30
Occupation: Software Engineer
Location: New York City
Hobbies: Hiking, Cooking, Traveling
Education: Bachelor's Degree in Computer Science
Experience: 5 years in software development, specializing in web applications and machine learning."""

# Summarization Function
def summarize(paragraph: str = test_para_1) -> str:
    model = pipeline("summarization", model="facebook/bart-large-cnn")
    response = model(paragraph)
    return response[0]['summary_text']

def text_generation(prompt: str) -> str:
    model = pipeline("text-generation", model="gpt2")
    response = model(prompt)
    return response[0]['generated_text']

def main():
    print(summarize())

if __name__ == "__main__":
    main()

