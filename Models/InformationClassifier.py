import os
from groq import Groq
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
# Use Groq as OpenAI-compatible endpoint

def classify_tweet_info_api(tweet: str) -> str:
    client = Groq(api_key="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")
    prompt = f"""
    You are a helpful assistant trained to classify tweets during disasters. 
    Given a tweet, respond with either "Informative" or "Not Informative",no other words,only either of just these two. 
    An informative tweet includes actionable information like needs, supplies, shelter, safety instructions, or calls for help. 
    A not informative tweet is vague, opinion-based, or irrelevant to disaster relief.

    Tweet: "{tweet}"
    Classification:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # or any other Groq-supported model like llama3-70b-8192
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=5,
        top_p=1.0,
        stream=False,

    )

    return response.choices[0].message.content


# print(classify_tweet("I have an exam tomorrow.Today was a disaster")) #not informative


model_path = "/Users/adhitya/Desktop/IDRIS/Backend/Models/ftmodel"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

id2label = {0: "not_informative", 1: "informative"}

def classify_tweet_info_custom(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    pred_id = torch.argmax(probs, dim=1).item()
    label = id2label[pred_id]
    confidence = probs[0][pred_id].item()
    return label

