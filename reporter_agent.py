from crewai import Agent
import json

reporter_agent = Agent(
    role="Research Reporter",
    goal="Aggregate TikTok sentiment findings into a structured report",
    backstory="Summarizes creator analysis into clear reports for decision-making.",
    verbose=True
)

def generate_report(analyzed_videos):
    summary = {
        "total_videos": len(analyzed_videos),
        "positive": len([v for v in analyzed_videos if "positive" in v["sentiment"].lower()]),
        "negative": len([v for v in analyzed_videos if "negative" in v["sentiment"].lower()]),
        "neutral": len([v for v in analyzed_videos if "neutral" in v["sentiment"].lower()])
    }
    return {"summary": summary, "details": analyzed_videos}
