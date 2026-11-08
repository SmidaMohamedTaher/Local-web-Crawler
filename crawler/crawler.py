from bs4 import BeautifulSoup
from collections import deque
from pathlib import Path


def parse_page(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, "html.parser")

    title = soup.title.text if soup.title else "No Title"

    text = soup.body.get_text(separator=" ", strip=True) if soup.body else ""

    links = [
        link.get("href")
        for link in soup.find_all("a")
        if link.get("href")
    ]

    return {
        "title": title,
        "text": text,
        "links": links
    }

def crawl(start_page):
    
    pages_data = []

    queue = deque([start_page])
    visited = set()

    while queue:

        current_page = queue.popleft()

        if current_page in visited:
            continue

        visited.add(current_page)

        result = parse_page(current_page)

        pages_data.append({
            "url": str(current_page),
            "title": result["title"],
            "text": result["text"],
            "links": result["links"]
        })


        for link in result["links"]:
            next_page = Path(link.lstrip("/"))

            if next_page.exists():
                next_page = str(next_page)

                if next_page not in visited:
                    queue.append(next_page)

    return pages_data
