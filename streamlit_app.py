
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="DocuMind | Research Assistant",
    page_icon="📄",
    layout="wide",
)

st.title("📄 DocuMind")
st.subheader("Your AI research-paper assistant")
st.write(
    "Upload a research paper, ask questions, "
    "and explore answers grounded in the document."
)

if "document_id" not in st.session_state:
    st.session_state.document_id = None

if "filename" not in st.session_state:
    st.session_state.filename = None

if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_file = st.file_uploader(
    "Upload a research paper (PDF)",
    type=["pdf"],
)

if st.button("Process PDF", disabled=uploaded_file is None):
    try:
        with st.spinner("Reading and indexing your paper..."):
            response = requests.post(
                f"{API_URL}/upload",
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "application/pdf",
                    )
                },
                timeout=300,
            )

        if response.ok:
            data = response.json()
            st.session_state.document_id = data["document_id"]
            st.session_state.filename = data["filename"]
            st.session_state.messages = []
            st.success(f"Successfully processed: {data['filename']}")
        else:
            st.error(f"Upload failed: {response.text}")

    except requests.RequestException as exc:
        st.error(
            "Could not connect to the DocuMind API. "
            "Make sure the FastAPI server is running."
        )
        st.caption(str(exc))

if st.session_state.document_id:
    st.divider()
    st.write(f"**Current paper:** {st.session_state.filename}")
    st.subheader("Ask your paper")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                st.caption("Sources")
                for source in message["sources"]:
                    st.write(
                        f"- {source['source']} — Page {source['page']}"
                    )

    question = st.chat_input("Ask a question about your paper...")

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Finding relevant passages and generating an answer..."):
                try:
                    response = requests.post(
                        f"{API_URL}/documents/"
                        f"{st.session_state.document_id}/ask",
                        json={"question": question},
                        timeout=180,
                    )

                    if response.ok:
                        data = response.json()
                        answer = data["answer"]
                        sources = data.get("sources", [])

                        st.markdown(answer)

                        if sources:
                            st.caption("Sources")
                            for source in sources:
                                st.write(
                                    f"- {source['source']} — "
                                    f"Page {source['page']}"
                                )

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer,
                                "sources": sources,
                            }
                        )
                    else:
                        error_message = (
                            f"Request failed ({response.status_code}): "
                            f"{response.text}"
                        )
                        st.error(error_message)

                except requests.RequestException:
                    st.error(
                        "Could not reach the API. "
                        "Check that the FastAPI server is running."
                    )
