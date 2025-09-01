from playwright.sync_api import sync_playwright

class TikTokScraper:
    def __init__(self, proxy=None):
        self.proxy = proxy

    def get_videos(self, profile_url, max_videos=10):
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, proxy={"server": self.proxy} if self.proxy else None)
            page = browser.new_page()
            page.goto(profile_url, wait_until="networkidle", timeout=60000)
            videos = page.query_selector_all("a[href*='/video/']")
            results = []
            for v in videos[:max_videos]:
                href = v.get_attribute("href")
                caption = v.inner_text()
                results.append({"url": href, "caption": caption})
            browser.close()
            return results
