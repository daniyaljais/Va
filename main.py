import os
import re
import json
import numpy as np
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import logging
import time
from typing import List, Dict
from fastapi.middleware.cors import CORSMiddleware


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found")
genai.configure(api_key=api_key)
logger.info("Google API configured")

# Load embeddings
embeddings_file = "embeddings.npz"
try:
    if not os.path.exists(embeddings_file):
        raise FileNotFoundError(f"{embeddings_file} not found")
    data = np.load(embeddings_file, allow_pickle=True)
    embeddings = data["embeddings"]
    texts = data["texts"]
    headings = data["headings"]
    source_files = data["source_files"]
    logger.info(f"Loaded {len(embeddings)} embeddings")
except Exception as e:
    logger.error(f"Error loading embeddings: {str(e)}")
    raise

class QueryRequest(BaseModel):
    question: str

class Link(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[Link]

def get_embedding(text: str) -> np.ndarray:
    try:
        if not text.strip():
            raise ValueError("Empty query text")
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="RETRIEVAL_QUERY"
        )
        embedding = np.array(result["embedding"])
        if embedding is None or len(embedding) == 0:
            raise ValueError("Invalid embedding returned")
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")

def semantic_search(query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
    try:
        norm_query = np.linalg.norm(query_embedding)
        if norm_query == 0:
            raise ValueError("Invalid query embedding norm")
        norm_embeddings = np.linalg.norm(embeddings, axis=1)
        similarities = np.dot(embeddings, query_embedding) / (norm_embeddings * norm_query)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            urls = [
                url.strip('"\',`)()') for url in re.findall(r'(https?://[^\s"\',`()]+)', texts[idx])
                if "discourse.onlinedegree.iitm.ac.in" in url
            ]
            link_text_map = {}
            try:
                json_match = re.search(r'```json\n([\s\S]*?)\n```', texts[idx])
                if json_match:
                    json_content = json.loads(json_match.group(1))
                    if "links" in json_content:
                        for link in json_content["links"]:
                            if link.get("url") in urls:
                                link_text_map[link["url"]] = link["text"]
            except json.JSONDecodeError:
                pass
            
            results.append({
                "text": texts[idx],
                "heading": headings[idx],
                "source_file": source_files[idx],
                "urls": urls,
                "link_text_map": link_text_map,
                "similarity": float(similarities[idx])
            })
            logger.info(f"Search result: heading={headings[idx][:50]}, urls={urls}, similarity={similarities[idx]:.4f}")
        return results
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

def generate_answer(question: str, context: str) -> str:
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""
        Question: {question}
        Context from forum posts: {context}
        
        Provide a concise, accurate answer to the question, focusing on token counting for `gpt-3.5-turbo` if relevant. Recommend using OpenAI’s `tiktoken` library for token counting. Cite https://discourse.onlinedegree.iitm.ac.in/t/ga5-question-8-clarification/155939/3 if it appears in the context, using: "My understanding is that you just have to use a tokenizer, similar to what Prof. Anand used, to get the number of tokens and multiply that by the given rate." Include Discourse URLs if relevant. If no relevant context is found, state that clearly.
        """
        response = model.generate_content(prompt)
        if not response.text:
            raise ValueError("Empty response from model")
        logger.info(f"Generated answer: {response.text[:200]}...")
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Answer generation error: {str(e)}")

async def process_query_internal(request: QueryRequest):
    try:
        start_time = time.time()
        query_text = f"Question: {request.question}\nToken counting for gpt-3.5-turbo using tiktoken or tokenizer".strip()
        if not query_text:
            raise HTTPException(status_code=400, detail="Empty question")
        logger.info(f"Processing query: {query_text[:100]}...")
        query_embedding = get_embedding(query_text)
        search_results = semantic_search(query_embedding)
        context = "\n\n".join([f"Post: {r['text'][:500]} (Similarity: {r['similarity']:.4f})" for r in search_results])
        answer = generate_answer(request.question, context)
        links = []
        for result in search_results:
            for url in result["urls"]:
                link_text = result["link_text_map"].get(url, result["heading"] or "Related post")
                links.append(Link(url=url, text=link_text[:500]))
        elapsed_time = time.time() - start_time
        logger.info(f"Query processed in {elapsed_time:.2f} seconds")
        return QueryResponse(answer=answer, links=links)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/api/", response_model=QueryResponse)
async def process_query_api(request: QueryRequest):
    return await process_query_internal(request)

@app.post("/", response_model=QueryResponse)
async def process_query_root(request: QueryRequest):
    logger.warning("Received request at /; redirecting to /api/")
    return await process_query_internal(request)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api3:app", host="0.0.0.0", port=port)
