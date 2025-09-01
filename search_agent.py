from crewai import Agent
from tools.serper_tool import SerperClient

search_agent = Agent(
    role="TikTok Researcher",
    goal="Find TikTok creator profiles related to the given subject",
    backstory="Expert in web search for social media profiles.",
    verbose=True
)

def search_creators(subject, country="us", max_results=5):
    client = SerperClient()
    return client.search_profiles(subject, country, max_results)
