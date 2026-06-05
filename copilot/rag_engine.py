import anthropic
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from copilot.prompts import CLAIMS_COPILOT_SYSTEM_PROMPT

Settings.embed_model = HuggingFaceEmbedding(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)

class ClaimsCopilot:
    def __init__(self, index_dir='data/knowledge_base_index/'):
        self.client = anthropic.Anthropic()
        self.model = 'claude-sonnet-4-5'
        storage_context = StorageContext.from_defaults(persist_dir=index_dir)
        self.index = load_index_from_storage(storage_context)
        self.retriever = self.index.as_retriever(similarity_top_k=3)
        self.messages = []  # conversation history
        self.sources = []   # source citations per turn

    def _retrieve_context(self, query: str) -> tuple:
        """Retrieve relevant chunks from the knowledge base."""
        nodes = self.retriever.retrieve(query)
        
        # Filter out low relevance chunks
        relevant_nodes = [n for n in nodes if n.score >= 0.3]
        
        # Fall back to top 1 if nothing passes the threshold
        if not relevant_nodes:
            relevant_nodes = nodes[:1]
        
        context_text = '\n\n'.join([n.text for n in relevant_nodes])
        sources = [{'text': n.text[:200], 'score': round(n.score, 3)}
                for n in relevant_nodes]
        return context_text, sources

    def chat(self, user_message: str) -> tuple:
        """Send a message and get a response grounded in the knowledge base."""
        # Step 1: retrieve relevant context
        context, sources = self._retrieve_context(user_message)

        # Step 2: build the system prompt with context injected
        system = CLAIMS_COPILOT_SYSTEM_PROMPT.format(context=context)

        # Step 3: add user message to history
        self.messages.append({'role': 'user', 'content': user_message})

        # Step 4: call Claude with full conversation history
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=self.messages,
        )

        # Step 5: add assistant response to history
        assistant_reply = response.content[0].text
        self.messages.append({'role': 'assistant', 'content': assistant_reply})
        self.sources.append(sources)

        return assistant_reply, sources

    def reset(self):
        """Clear conversation history."""
        self.messages = []
        self.sources = []