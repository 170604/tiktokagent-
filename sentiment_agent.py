from crewai import Agent
from vertexai.preview import generative_models as gm
import os

sentiment_agent = Agent(
    role="Sentiment Analyst",
    goal="Analyze TikTok video captions and classify sentiment",
    backstory="Uses Google Gemini to perform NLP sentiment classification.",
    verbose=True
)

def analyze_sentiment(videos):
    gemini = gm.GenerativeModel("gemini-1.5-flash")
    analyzed = []
    for v in videos:
        prompt = f"Classify sentiment of this TikTok caption: '{v['caption']}'"
        resp = gemini.generate_content(prompt)
        analyzed.append({**v, "sentiment": resp.text})
    return analyzed
