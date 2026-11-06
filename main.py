from crawler.crawler import crawl
import json

pages_data = crawl("pages\\c.html")

# for page in pages_data:
#     print(page)
#     print()

file = open("crawled_pages.json", "w", encoding="utf-8")

json.dump(pages_data, file, ensure_ascii=False, indent=4)

file.close()

print(len(pages_data))