from crewai import Agent, Task, Crew
from agents.search_agent import search_agent
from agents.scraper_agent import scraper_agent
from agents.sentiment_agent import sentiment_agent
from agents.reporter_agent import reporter_agent

def run(subject: str, max_creators: int = 5, max_videos: int = 10, country: str = "us"):
    # Define tasks
    search_task = Task(
        agent=search_agent,
        description=f"Find {max_creators} TikTok creators related to {subject} in {country}",
        expected_output="List of TikTok profile URLs"
    )

    scrape_task = Task(
        agent=scraper_agent,
        description="Scrape profile pages and extract video metadata",
        expected_output="List of video URLs + captions"
    )

    sentiment_task = Task(
        agent=sentiment_agent,
        description="Perform sentiment analysis on TikTok video captions",
        expected_output="List of videos with Positive/Neutral/Negative sentiment"
    )

    report_task = Task(
        agent=reporter_agent,
        description="Generate structured summary report",
        expected_output="JSON + text report"
    )

    # Orchestrate
    crew = Crew(tasks=[search_task, scrape_task, sentiment_task, report_task])
    return crew.run()

if __name__ == "__main__":
    result = run(subject="trading", max_creators=3, max_videos=5)
    print(result)
