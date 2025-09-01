import os, httpx

class SerperClient:
    def __init__(self):
        self.api_key = os.getenv("SERPER_API_KEY")
        if not self.api_key:
            raise RuntimeError("SERPER_API_KEY not set")
        self.endpoint = "https://google.serper.dev/search"

    def search_profiles(self, subject, country="us", max_results=5):
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        payload = {"q": f'site:tiktok.com "{subject}"', "gl": country, "num": max_results}
        r = httpx.post(self.endpoint, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        return [item["link"] for item in data.get("organic", []) if "tiktok.com" in item["link"]]
