import streamlit as st
import pandas as pd
import re

from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

st.set_page_config(page_title="Receipt RAG Checker", layout="wide")

st.title("Receipt RAG Checker")
st.write(
    "This app checks whether a receipt is saved in the database using "
    "LlamaIndex, embeddings, vector retrieval, top-k search, grouped context, "
    "and a grounded response."
)

@st.cache_resource
def load_index():
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    df = pd.read_csv("receipts.csv")

    documents = []

    for _, row in df.iterrows():
        text = (
            f"Receipt ID: {row['receipt_id']}. "
            f"Customer: {row['customer']}. "
            f"Amount: {row['amount']}. "
            f"Date: {row['date']}. "
            f"Status: {row['status']}. "
            f"Description: {row['description']}."
        )

        documents.append(
            Document(
                text=text,
                metadata={
                    "receipt_id": row["receipt_id"],
                    "customer": row["customer"],
                    "status": row["status"],
                    "date": row["date"]
                }
            )
        )

    index = VectorStoreIndex.from_documents(documents)
    return df, index


df, index = load_index()

TOP_K = 3
retriever = index.as_retriever(similarity_top_k=TOP_K)


def extract_receipt_id(question):
    match = re.search(r"R\d+", question.upper())
    return match.group() if match else None


def exact_db_lookup(receipt_id):
    result = df[df["receipt_id"] == receipt_id]
    if result.empty:
        return None
    return result.iloc[0].to_dict()


def build_grouped_context(nodes):
    grouped_context = {}

    for node in nodes:
        receipt_id = node.metadata.get("receipt_id", "unknown")
        grouped_context.setdefault(receipt_id, []).append(node.text)

    return grouped_context


def generate_grounded_answer(grouped_context):
    if not grouped_context:
        return "I do not have enough retrieved context to answer."

    answer = "Based on the retrieved receipt context:\n\n"

    for receipt_id, chunks in grouped_context.items():
        answer += f"Receipt ID: {receipt_id}\n"
        for chunk in chunks[:3]:
            answer += f"- {chunk}\n"
        answer += "\n"

    answer += (
        "This response is grounded only in the retrieved receipt records. "
        "For production, an LLM can be connected with temperature=0.0 or 0.2."
    )

    return answer


question = st.text_input(
    "Ask a receipt question",
    value="Is receipt R1001 saved in the DB?"
)

if st.button("Run RAG Check"):
    receipt_id = extract_receipt_id(question)

    st.subheader("1. Exact DB Lookup")

    if receipt_id:
        exact_result = exact_db_lookup(receipt_id)

        if exact_result:
            st.success(f"Exact match found for {receipt_id}")
            st.json(exact_result)
        else:
            st.error(f"No exact DB record found for {receipt_id}")
    else:
        st.info("No exact receipt ID found. Using vector retrieval.")

    st.subheader("2. Top-k Vector Retrieval")

    retrieved_nodes = retriever.retrieve(question)
    grouped_context = build_grouped_context(retrieved_nodes)

    st.write(f"Top-k retrieval value: {TOP_K}")

    for receipt_group, chunks in grouped_context.items():
        with st.expander(f"Receipt Group: {receipt_group}"):
            for chunk in chunks:
                st.write(chunk)

    st.subheader("3. Context Window")

    selected_context = []
    for _, chunks in grouped_context.items():
        selected_context.extend(chunks)

    selected_context = selected_context[:3]

    st.write("Using top 3 retrieved chunks as context window:")
    for context in selected_context:
        st.write("- " + context)

    st.subheader("4. Grounded Answer")

    answer = generate_grounded_answer(grouped_context)
    st.write(answer)

    st.subheader("5. Evaluation Plan")

    st.write(
        """
        Evaluation metrics/tools planned:
        - Precision@K
        - Recall@K
        - MRR
        - DeepEval for LLM regression tests
        - RAGAs for faithfulness, answer relevancy, contextual precision, and contextual recall
        - TruLens for observability, latency, token usage, and feedback tracking
        """
    )
