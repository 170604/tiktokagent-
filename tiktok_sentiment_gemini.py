#!/usr/bin/env python3
"""
TikTok Sentiment & Virality Analyzer — Minimal Working Build (Gemini + Serper + httpx/Playwright)

Usage example:
  setx SERPER_API_KEY "YOUR_SERPER_KEY"
  setx GEMINI_API_KEY "YOUR_GEMINI_KEY"
  # new PowerShell after setx, or use $env:SERPER_API_KEY / $env:GEMINI_API_KEY for current shell

  pip install -U google-generativeai httpx[http2] pydantic bs4 loguru tenacity orjson rapidfuzz playwright
  python -m playwright install

  python tiktok_sentiment_gemini.py --subject "trading" --max-creators 5 --max-videos 10 --use-playwright
"""

from __future__ import annotations
import argparse, os, re, json, time, sys, math
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict


from loguru import logger


import httpx
from bs4 import BeautifulSoup


from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


try:
    import orjson
    def json_dumps(obj: Any) -> str:
        return orjson.dumps(obj, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS).decode("utf-8")
except Exception:
    def json_dumps(obj: Any) -> str:
        return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)


import google.generativeai as genai

class GeminiLLM:
    def __init__(self, model: str = "gemini-1.5-flash", temperature: float = 0.2):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY in env")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        resp = self.model.generate_content(
            prompt,
            generation_config={"temperature": self.temperature}
        )
        return getattr(resp, "text", "") or ""



SERPER_ENDPOINT = "https://google.serper.dev/search"

class SerperClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SERPER_API_KEY")
        if not self.api_key:
            raise RuntimeError("Set SERPER_API_KEY in env")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=6),
           retry=retry_if_exception_type((httpx.HTTPError,)))
    def search(self, q: str, country: str = "us") -> Dict[str, Any]:
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        payload = {"q": q, "gl": country.lower()}
        with httpx.Client(timeout=30) as client:
            r = client.post(SERPER_ENDPOINT, headers=headers, json=payload)
            r.raise_for_status()
            return r.json()



DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

def make_client(proxy: Optional[str] = None) -> httpx.Client:
    if proxy:
        return httpx.Client(timeout=40, headers=DEFAULT_HEADERS, follow_redirects=True, http2=True, transport=httpx.HTTPTransport(proxy=proxy))
    return httpx.Client(timeout=40, headers=DEFAULT_HEADERS, follow_redirects=True, http2=True)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10),
       retry=retry_if_exception_type((httpx.HTTPError,)))
def fetch(url: str, client: httpx.Client) -> str:
    r = client.get(url)
    r.raise_for_status()
    return r.text

def extract_video_links_from_html(html: str) -> List[str]:
    # Look for absolute & relative TikTok video URLs
    abs_urls = re.findall(r'https?://www\.tiktok\.com/@[^/"]+/video/\d+', html)
    rel_urls = re.findall(r'/@[^/"]+/video/\d+', html)
    # Normalize
    rel_full = [f"https://www.tiktok.com{u}" for u in rel_urls]
    urls = list(dict.fromkeys(abs_urls + rel_full))  # dedupe preserving order
    return urls

def extract_caption_title(html: str) -> Tuple[Optional[str], Optional[str]]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.text.strip() if soup.title else None
    # OG description is usually the caption + hashtags
    og_desc = None
    og = soup.find("meta", attrs={"property": "og:description"})
    if og and og.get("content"):
        og_desc = og["content"].strip()
    return og_desc, title



@dataclass
class VideoInfo:
    url: str
    caption: Optional[str] = None
    page_title: Optional[str] = None
    sentiment: Optional[str] = None   # positive | negative | mixed | neutral
    virality_reasons: Optional[str] = None

@dataclass
class CreatorInfo:
    username: str
    profile_url: str
    videos: List[VideoInfo]

@dataclass
class FlowResult:
    subject: str
    creators: List[CreatorInfo]
    country: str
    generated_at: float



class TikTokFlow:
    def __init__(self,
                 subject: str,
                 max_creators: int = 5,
                 max_videos: int = 10,
                 country: str = "us",
                 proxy: Optional[str] = None,
                 use_playwright: bool = False):
        self.subject = subject
        self.max_creators = max_creators
        self.max_videos = max_videos
        self.country = country
        self.proxy = proxy
        self.use_playwright = use_playwright

        self.search = SerperClient()
        self.llm = GeminiLLM()
        self.http = make_client(proxy)


    def find_creator_profiles(self) -> List[str]:
      
        q = f'site:tiktok.com "@{self.subject}" OR "{self.subject}" site:tiktok.com/@'
        logger.info(f"Searching creators with Serper: {q}")
        data = self.search.search(q=q, country=self.country)
        links = []
        for item in (data.get("organic") or []):
            url = item.get("link") or ""
            if "tiktok.com/@" in url:
                links.append(url.split("?")[0])
      
        seen = set()
        profiles = []
        for u in links:
            m = re.search(r'https?://www\.tiktok\.com/(@[^/?#]+)', u)
            if not m:
                continue
            username = m.group(1)
            if username in seen:
                continue
            seen.add(username)
            profiles.append(f"https://www.tiktok.com/{username}")
            if len(profiles) >= self.max_creators:
                break
        logger.info(f"Found {len(profiles)} profile(s)")
        return profiles

 
    def get_html(self, url: str) -> str:
        if not self.use_playwright:
            return fetch(url, self.http)
        
        from playwright.sync_api import sync_playwright
        logger.debug(f"Playwright fetching: {url}")
        launch_args = {"headless": True}
        if self.proxy:
           
            launch_args["args"] = [f"--proxy-server={self.proxy}"]
        with sync_playwright() as p:
            browser = p.chromium.launch(**launch_args)
            try:
                ctx = browser.new_context(user_agent=DEFAULT_HEADERS["User-Agent"], locale="en-US")
                page = ctx.new_page()
                page.goto(url, wait_until="networkidle", timeout=60000)
               
                page.wait_for_timeout(1500)
                html = page.content()
                return html
            finally:
                browser.close()


    def scrape_creator_videos(self, profile_url: str) -> List[VideoInfo]:
        logger.info(f"Scraping profile: {profile_url}")
        html = self.get_html(profile_url)
        video_links = extract_video_links_from_html(html)
        if not video_links:
            logger.warning("No video links found on profile page.")
            return []

        kept = []
        for url in video_links:
            if len(kept) >= self.max_videos:
                break
            try:
                vhtml = self.get_html(url)
                og_desc, title = extract_caption_title(vhtml)
                kept.append(VideoInfo(url=url, caption=og_desc, page_title=title))
            except Exception as e:
                logger.warning(f"Video fetch failed: {url} -> {e}")
        logger.info(f"Collected {len(kept)} video(s) for {profile_url}")
        return kept

    # ---- LLM sentiment & virality
    def analyze_video(self, v: VideoInfo) -> VideoInfo:
        text = v.caption or v.page_title or ""
        if not text:
            v.sentiment = "unknown"
            v.virality_reasons = "No caption/title available."
            return v
        prompt = f"""
You are a strict social-media analyst.

Given this TikTok video text, classify sentiment and explain virality briefly.

Text:
\"\"\"{text[:2000]}\"\"\"

Return in this exact JSON with keys: sentiment (one of: positive, negative, mixed, neutral), reasons (<=80 words).
"""
        try:
            resp = self.llm.generate(prompt)
            # try to parse JSON from the response; if not JSON, fall back to heuristics
            sent, reasons = None, None
            # crude JSON scrape
            m = re.search(r'\{.*\}', resp, flags=re.S)
            if m:
                obj = json.loads(m.group(0))
                sent = (obj.get("sentiment") or "").lower()
                reasons = obj.get("reasons")
            if not sent:
                # fallback heuristic
                sent = "mixed"
                reasons = resp.strip()[:400]
            v.sentiment = sent
            v.virality_reasons = reasons
        except Exception as e:
            v.sentiment = "error"
            v.virality_reasons = f"Analysis failed: {e}"
        return v

 
    def kickoff(self) -> Dict[str, Any]:
        creators_urls = self.find_creator_profiles()
        creators: List[CreatorInfo] = []
        for purl in creators_urls:
            uname = re.search(r'https?://www\.tiktok\.com/(@[^/?#]+)', purl)
            username = uname.group(1) if uname else purl
            videos = self.scrape_creator_videos(purl)
            analyzed: List[VideoInfo] = []
            for v in videos:
                analyzed.append(self.analyze_video(v))
            creators.append(CreatorInfo(username=username, profile_url=purl, videos=analyzed))
        result = FlowResult(
            subject=self.subject,
            creators=creators,
            country=self.country,
            generated_at=time.time(),
        )
       
        out = {
            "subject": result.subject,
            "country": result.country,
            "generated_at": result.generated_at,
            "creators": [
                {
                    "username": c.username,
                    "profile_url": c.profile_url,
                    "videos": [asdict(v) for v in c.videos]
                }
                for c in result.creators
            ],
        }
        return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TikTok Sentiment & Virality Analyzer (Gemini + Serper)")
    p.add_argument("--subject", required=True, help="Subject to analyze")
    p.add_argument("--max-creators", type=int, default=5)
    p.add_argument("--max-videos", type=int, default=10)
    p.add_argument("--country", default="us")
    p.add_argument("--proxy", default=None, help="http://host:port or socks5://host:port")
    p.add_argument("--use-playwright", action="store_true", help="Render JS via Playwright (more reliable)")
    p.add_argument("--out", default="report.json")
    return p

def main():
    args = build_argparser().parse_args()
    if args.proxy:
        os.environ["HTTP_PROXY"] = args.proxy
        os.environ["HTTPS_PROXY"] = args.proxy

    flow = TikTokFlow(
        subject=args.subject,
        max_creators=args.max_creators,
        max_videos=args.max_videos,
        country=args.country,
        proxy=args.proxy,
        use_playwright=args.use_playwright,
    )
    result = flow.kickoff()
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(json_dumps(result))
    print(f"Saved: {args.out}")
    print(json_dumps(result)[:2000])

if __name__ == "__main__":
    main()
