import networkx as nx
import matplotlib.pyplot as plt
import json

class WebGraph:

    def __init__(self, pages_data):

        self.graph = nx.DiGraph()

        
        for page in pages_data:

            url = self.normalize_path(page["url"])

            self.graph.add_node(url)

        for page in pages_data:

            source = self.normalize_path(page["url"])

            for link in page["links"]:

                target = self.normalize_path(link)

                if target in self.graph:
                    self.graph.add_edge(source, target)

    def normalize_path(self, path):

        return path.replace("\\", "/").lstrip("/")

    def get_graph(self):
        return self.graph
    
    def calculate_pagerank(self):

        self.pagerank_scores = nx.pagerank(
            self.graph,
            alpha=0.85
        )

        with open("APIs/pagerank_scores.json", "w", encoding="utf-8") as file:
            json.dump(self.pagerank_scores, file, ensure_ascii=False, indent=4)
        return self.pagerank_scores

# with open("APIs/crawled_pages.json", "r", encoding="utf-8") as file:
#      pages_data = json.load(file)

# w = WebGraph(pages_data=pages_data)

# w.calculate_pagerank()