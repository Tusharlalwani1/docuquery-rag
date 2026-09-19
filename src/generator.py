import os
from groq import Groq

class RAGGenerator:
    def __init__(self, model: str = "openai/gpt-oss-20b"):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = model

    def stream_answer(self, query: str, context_chunks: list[str]):
        if not context_chunks:
            yield "I could not find relevant information in the provided documents."
            return

        context_str = "\n---\n".join(context_chunks)
        system_prompt = f"""
You are a precise technical documentation assistant.
Answer the question using ONLY the provided context snippets.
If the context does not contain the answer, explicitly state that you don't know based on the files.

CONTEXT:
{context_str}
"""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content