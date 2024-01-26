from typing import Callable, Dict, Any
from tool_library.tools import *
from tool_library.utils import *
from tool_library.factory import FastApiToolFactory
from tool_library.events import EventLog

# Similarity:
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from sentence_transformers import util


NEW_TYPE = "New Tool Added"
NEW_MESSAGE = "You can now use this tool, use find_tools to find out how to use it."

REMOVED_TOOL = "Tool Removed"
REMOVED_MESSAGE = "You can no longer use this tool."


class ToolLibrary:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.tools = {}
        self.model = SentenceTransformer(
            embedding_model
        )  # Initialize the sentence transformer model
        self.embeddings = {}  # Dictionary to store tool embeddings
        self.event_log = EventLog()

    def _create_embeddings(self, tool: Tool):
        # Create embeddings for name and description

        props = [tool.name, tool.description]
        embeddings = self.model.encode(props)
        return embeddings

    def _update_embeddings_index(self, name: str, embeddings):
        # Update the embeddings index
        self.embeddings[name] = embeddings

    def register_tool_embeddings(self, tool):
        # Update embeddings for the new tool
        embeddings = self._create_embeddings(tool)
        self._update_embeddings_index(tool.name, embeddings)

    def register_tool(self, name: str, description: str, function: Callable):
        if name in self.tools:
            raise ValueError(f"A tool with the name '{name}' is already registered.")

        if is_ray_remote_function(function):
            self.tools[name] = RayTool(name, description, function)
        else:
            self.tools[name] = Tool(name, description, function)

        self.event_log.add_event(NEW_TYPE, name, NEW_MESSAGE)
        self.register_tool_embeddings(self.tools[name])

    def register_api_tool(self, service_url="http://127.0.0.1:8000", tool_routes=None, api_key=None):
        belt = FastApiToolFactory(service_url=service_url, tool_routes=tool_routes, api_key=api_key)
        belt.introspect_service()

        for t in belt.tools:
            self.tools[t.name] = t
            self.register_tool_embeddings(self.tools[t.name])
            self.event_log.add_event(
                event_type=NEW_TYPE, tool=t.name, message=NEW_MESSAGE
            )
    
    def execute_tool(self, tool_name: str, params: Dict) -> Any:
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found.")
        return self.tools[tool_name].execute(params)

    def find_tools(self, query: str, top_k: int = 3) -> List[str]:
        # Create a flat list of all embeddings and a corresponding mapping to tool names
        corpus_embeddings = []
        corpus = []

        for name, embeddings in self.embeddings.items():
            corpus_embeddings.extend(embeddings)
            corpus.extend([name] * len(embeddings))

        # Convert to a 2D tensor where each row is an embedding
        corpus_embeddings = torch.tensor(corpus_embeddings)
        if len(corpus_embeddings.shape) == 1:
            # If the embeddings are 1D, reshape to 2D (this handles the case of a single embedding)
            corpus_embeddings = corpus_embeddings.unsqueeze(0)

        query_embedding = self.model.encode(query, convert_to_tensor=True)
        if len(query_embedding.shape) == 1:
            # Reshape query_embedding to 2D if necessary
            query_embedding = query_embedding.unsqueeze(0)

        # Compute similarity
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_k = min(top_k, len(corpus))
        top_results = torch.topk(cos_scores, k=top_k)

        # Retrieve and map top matches back to tool names
        results = []
        for score, idx in zip(top_results[0], top_results[1]):
            if score > 0.5:  # Threshold for similarity
                tool_name = corpus[idx]
                if tool_name not in [
                    result["name"] for result in results
                ]:  # Remove duplicates
                    results.append(
                        {"name": tool_name, "params": self.tools[tool_name].params}
                    )

        return results

    def get_events(self, minutes_ago=0):
        return self.event_log.find_events(minutes_ago)

    def get_tools(self) -> Dict:
        return self.tools

    def remove_tool(self, name):
        if name in self.tools:
            del self.tools[name]
            del self.embeddings[name]
            event_log.add_event(REMOVED_TOOL, name, REMOVED_MESSAGE)
            return True
        return False

    def get_tool_stats(self, tool_name: str) -> Dict:
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found.")
        tool = self.tools[tool_name]
        stats = tool.stats
        return {
            "creation_time": stats.creation_time,
            "call_count": stats.call_count,
            "average_execution_time": stats.average_execution_time,
        }
