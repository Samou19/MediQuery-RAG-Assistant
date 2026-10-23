"""MediQuery — Modern medical RAG assistant."""

import logging

import streamlit as st
from langchain_core.documents import Document

from mediquery.config import ANTHROPIC_API_KEY, APP_TITLE, SOURCE_PDF
from mediquery.generation.generator import MediQueryGenerator
from mediquery.pipeline import MediQueryPipeline
from mediquery.retrieval.hybrid import HybridRetriever
from mediquery.retrieval.pipeline_retriever import HybridPipelineRetriever
from mediquery.retrieval.vector_store import get_vector_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root {
        --mq-blue: #2563eb;
        --mq-dark: #142747;
        --mq-muted: #64748b;
        --mq-border: #e2eaf4;
        --mq-bg: #f6f9fd;
    }
    .stApp { background: var(--mq-bg); font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Manrope', sans-serif !important; color: var(--mq-dark); }
    [data-testid="stSidebar"] { background: #fff; border-right: 1px solid var(--mq-border); }
    [data-testid="stSidebar"] .block-container { padding-top: 1.8rem; }
    .mq-brand { display: flex; align-items: center; gap: 12px; margin-bottom: 2rem; }
    .mq-logo {
        display: flex; align-items: center; justify-content: center;
        width: 48px; height: 48px; border-radius: 15px;
        background: linear-gradient(135deg, #2563eb, #60a5fa);
        color: white; font-size: 25px; box-shadow: 0 7px 18px #2563eb30;
    }
    .mq-brand-name { font-family: 'Manrope', sans-serif; font-size: 1.35rem; font-weight: 800; color: var(--mq-dark); line-height: 1.2; }
    .mq-brand-subtitle { color: var(--mq-muted); font-size: .78rem; margin-top: 4px; }
    .mq-hero {
        background: linear-gradient(125deg, #e8f2ff 0%, #f4f8ff 60%, #fff 100%);
        border: 1px solid #dbeafe; border-radius: 24px; padding: 2rem 2.1rem; margin-bottom: 1.5rem;
    }
    .mq-eyebrow {
        color: var(--mq-blue);
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        text-transform: none;
        line-height: 1.3;
        margin-bottom: 1rem;
    }
    .mq-hero h1 { font-size: clamp(2.1rem, 4vw, 3.1rem); line-height: 1.1; margin: 0 0 .8rem 0; letter-spacing: -.045em; }
    .mq-hero p { color: #475d7b; max-width: 720px; line-height: 1.7; margin-bottom: 0; }
    .mq-feature-row { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 1.25rem; }
    .mq-feature { background: #ffffffb8; border: 1px solid #dbeafe; border-radius: 999px; padding: 7px 12px; color: #24436e; font-size: .82rem; }
    .mq-section-title { font-family: 'Manrope', sans-serif; font-size: 1.12rem; font-weight: 750; color: var(--mq-dark); margin: 1.2rem 0 .8rem 0; }
    .mq-card { background: white; border: 1px solid var(--mq-border); border-radius: 18px; padding: 1.3rem 1.4rem; margin: .6rem 0 1rem 0; box-shadow: 0 5px 20px #18315306; }
    .mq-answer-label { color: #087443; background: #ecfdf3; border: 1px solid #bbf7d0; padding: 9px 12px; border-radius: 10px; font-size: .85rem; font-weight: 700; margin-bottom: 1rem; }
    .mq-disclaimer { background: #eff6ff; border: 1px solid #dbeafe; border-radius: 14px; padding: 14px; color: #355174; font-size: .8rem; line-height: 1.6; }
    .mq-footer { text-align: center; color: #94a3b8; font-size: .78rem; padding: 2rem 0 .5rem 0; }
    div.stButton > button { border-radius: 11px; font-weight: 600; min-height: 42px; transition: all .15s ease; }
    div.stButton > button[kind="primary"] { background: var(--mq-blue); border: 1px solid var(--mq-blue); }
    div.stButton > button[kind="primary"]:hover { background: #1d4ed8; border-color: #1d4ed8; }
    div[data-testid="stForm"] { background: white; border: 1px solid var(--mq-border); border-radius: 18px; padding: 1.25rem; }
    @media (max-width: 700px) { .mq-hero { padding: 1.3rem; border-radius: 17px; } .mq-card { padding: 1rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_pipeline() -> MediQueryPipeline:
    """Load the existing Chroma collection and construct the RAG pipeline."""
    vector_store = get_vector_store()
    stored_data = vector_store.get(include=["documents", "metadatas"])
    contents = stored_data.get("documents") or []
    metadatas = stored_data.get("metadatas") or []

    documents = [
        Document(page_content=content, metadata=metadata or {})
        for content, metadata in zip(contents, metadatas, strict=True)
        if content and content.strip()
    ]
    if not documents:
        raise RuntimeError(
            "Aucun passage trouvé dans ChromaDB. Vérifie l'indexation des documents."
        )

    hybrid = HybridRetriever(documents)
    retriever = HybridPipelineRetriever(hybrid)
    generator = MediQueryGenerator()
    return MediQueryPipeline(retriever=retriever, generator=generator)


with st.sidebar:
    st.markdown(
        """
        <div class="mq-brand">
            <div class="mq-logo">✚</div>
            <div>
                <div class="mq-brand-name">MediQuery</div>
                <div class="mq-brand-subtitle">Assistant médical RAG</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        ["Questionner", "À propos"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("#### 📚 Document de référence")

    st.markdown(
        "[📄 Consulter le guide PDF en ligne](https://www.has-sante.fr/upload/docs/application/pdf/2025-07/parcours_de_soins_du_patient_adulte_vivant_avec_un_diabete_de_type_2_-_guide_2025-07-15_16-42-8_797.pdf)",
        unsafe_allow_html=False,
    )

    st.divider()
    st.markdown(
        """
        <div class="mq-disclaimer">
            <strong>ⓘ Usage médical</strong><br><br>
            MediQuery explique le contenu des documents fournis.
            Il ne remplace ni un professionnel de santé, ni un diagnostic,
            ni un avis médical personnalisé.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("MediQuery · Prototype de recherche médicale")

    st.markdown(
        """
        <div style="
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid #e2e8f0;
            text-align: center;
            color: #64748b;
            font-size: 0.8rem;
            line-height: 1.6;
        ">
            <div style="font-weight: 600; color: #334155;">
                Développé par Sambou Diao
            </div>
            <div>Data &amp; AI Engineer</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if page == "Questionner":
    st.markdown(
        """
        <div class="mq-hero">
            <div class="mq-eyebrow">
                Assistant médical basé sur vos documents
            </div>
            <p>
                Posez vos questions sur le parcours de soins du diabète de type 2.
                MediQuery recherche les passages pertinents et prépare une réponse
                fondée sur le document de référence.
            </p>
            <div class="mq-feature-row">
                <span class="mq-feature">✓ Réponses documentées</span>
                <span class="mq-feature">⌕ Recherche hybride</span>
                <span class="mq-feature">▤ Sources consultables</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    example_questions = [
        "Quelles sont les étapes du parcours de soins du diabète de type 2 ?",
        "Comment dépister et diagnostiquer un diabète de type 2 ?",
        "Quels sont les principes du traitement ?",
        "Quelles complications faut-il rechercher ?",
    ]

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None

    def select_example(question: str) -> None:
        """Queue a suggested question for processing on the next rerun."""
        st.session_state.pending_question = question

    # Show suggested questions before the first exchange.
    if not st.session_state.messages:
        st.markdown("### Quelques idées pour commencer")
        cols = st.columns(2)
        for index, example in enumerate(example_questions):
            with cols[index % 2]:
                st.button(
                    example,
                    key=f"example_{index}",
                    use_container_width=True,
                    on_click=select_example,
                    args=(example,),
                )

    # Conversation history.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("📖 Sources documentaires"):
                    for source in message["sources"]:
                        page_value = source.get("page")
                        page_label = str(page_value) if page_value is not None else "non précisée"
                        st.markdown(
                            f"- **{source['document_name']}** — page {page_label}"
                        )

    # Chat input stays at the bottom of the Streamlit page.
    typed_question = st.chat_input(
        "Posez votre question sur le parcours de soins...",
        max_chars=1000,
    )
    question = st.session_state.pending_question or typed_question
    st.session_state.pending_question = None

    if question:
        normalized_question = question.strip()
        if not normalized_question:
            st.warning("Saisissez une question avant de continuer.")
        elif not ANTHROPIC_API_KEY:
            st.error(
                "La clé API Anthropic n'est pas configurée. Ajoutez "
                "ANTHROPIC_API_KEY dans votre fichier .env, puis redémarrez l'application."
            )
        else:
            st.session_state.messages.append(
                {"role": "user", "content": normalized_question}
            )
            with st.chat_message("user"):
                st.markdown(normalized_question)

            try:
                with st.chat_message("assistant"):
                    with st.spinner(
                        "Recherche documentaire et préparation de votre réponse…"
                    ):
                        pipeline = load_pipeline()
                        response = pipeline.run(normalized_question)

                    st.markdown(response.answer)

                    # N'afficher les sources que si la réponse n'est pas une abstention
                    source_items = []
                    if not response.abstained and response.sources:
                        source_items = [
                            {
                                "document_name": source.document_name,
                                "page": source.page,
                                "section": source.section,
                                "chunk_id": source.chunk_id,
                            }
                            for source in response.sources
                        ]

                        with st.expander("📖 Sources documentaires"):
                            for source in source_items:
                                page_value = source["page"]
                                page_label = (
                                    str(page_value)
                                    if page_value is not None
                                    else "non précisée"
                                )
                                st.markdown(
                                    f"- **{source['document_name']}** "
                                    f"— page {page_label}"
                                )


                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response.answer,
                        "sources": source_items,
                    }
                )
            except Exception as exc:
                logger.exception("Erreur pendant l'exécution du pipeline.")
                error_message = (
                    f"Le traitement a échoué ({type(exc).__name__}). "
                    "Vérifiez le terminal pour consulter les détails."
                )
                with st.chat_message("assistant"):
                    st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message, "sources": []}
                )

elif page == "À propos":
    st.markdown(
        """
        <div class="mq-hero">
            <div class="mq-eyebrow">À propos du projet</div>
            <h1>MediQuery</h1>
            <p>
                Un assistant RAG qui associe recherche documentaire et génération
                de langage pour faciliter l'exploration de documents médicaux.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Comment fonctionne MediQuery ?")
    st.markdown(
        """
        **1. Recherche hybride**  
        BM25 identifie les correspondances lexicales et la recherche vectorielle
        retrouve les passages sémantiquement proches. RRF fusionne les classements.

        **2. Construction du contexte**  
        Les passages récupérés sont transmis au modèle avec leurs métadonnées.

        **3. Génération augmentée par récupération (RAG)**  
        Claude formule une réponse à partir du contexte documentaire fourni.

        **4. Traçabilité**  
        Les références des passages récupérés sont affichées pour faciliter
        la vérification des réponses.
        """
    )
    st.markdown("### Technologies")
    st.markdown("`Python` · `Streamlit` · `LangChain` · `ChromaDB` · `Hugging Face` · `BM25` · `RRF` · `Anthropic Claude`")
    st.info(
        "MediQuery est un prototype de démonstration. Il ne constitue pas un dispositif "
        "médical et ne remplace pas l'évaluation d'un professionnel de santé."
    )

st.markdown(
    '<div class="mq-footer">MediQuery · Assistant RAG médical · Prototype de démonstration</div>',
    unsafe_allow_html=True,
)
