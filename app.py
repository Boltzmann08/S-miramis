import streamlit as st
import os
import fitz
import base64
from PIL import Image
import openai
from typing import List, Any
import uuid
import re

from haystack import Pipeline, Document
from haystack.utils import Secret
from haystack.dataclasses import ChatMessage
from haystack import component
from haystack.core.component.types import Variadic
from itertools import chain

# Haystack components
from haystack_experimental.chat_message_stores.in_memory import InMemoryChatMessageStore
from haystack_experimental.components.retrievers import ChatMessageRetriever
from haystack_experimental.components.writers import ChatMessageWriter
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from haystack.components.embedders import OpenAITextEmbedder,OpenAIDocumentEmbedder
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack.components.generators import OpenAIGenerator
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.builders import ChatPromptBuilder, PromptBuilder
from haystack.components.converters import OutputAdapter
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.components.converters import TextFileToDocument


######################################
# Extraction des infos des documents patients
######################################

def extract_images_from_pdf(pdf_path, output_folder):
    """Extrait chaque page du PDF sous forme d'image."""
    os.makedirs(output_folder, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_paths = []

    for i, page in enumerate(doc):
        pix = page.get_pixmap()
        img_path = os.path.join(output_folder, f"page_{i+1}.png")
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img.save(img_path)
        image_paths.append(img_path)

    return image_paths

def encode_image_to_base64(image_path):
    """Encode une image en base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def process_image_with_openai(image_path):
    """Utilise OpenAI Vision API pour extraire et structurer les informations médicales d'une image."""
    image_base64 = encode_image_to_base64(image_path)

    response = openai.chat.completions.create(
        model="chatgpt-4o-latest",
        messages=[
            {"role": "system", "content": "Vous êtes un assistant avancé spécialisé dans l'OCR et la structuration de documents médicaux. Vous êtes un expert en traitement et structuration de documents médicaux. Votre tâche est d’extraire, organiser et structurer les informations contenues dans le document fourni de manière rigoureuse et exploitable dans un pipeline RAG."},
            {"role": "user", "content": [
                {"type": "text", "text": "### Consignes d'extraction et de structuration :" 
                 "1.Extraire l'intégralité des données" "médicales sans altération ni omission."
                 "2..Structurer les informations sous forme de texte clair et segmenté avec des titres et sous-titres explicites."
                 "3.Distinguer et organiser les sections suivantes si elles sont présentes dans le document exemple:"
                 "a.Informations Générales du Patient : Nom, âge, sexe, antécédents médicaux, traitements en cours."
                 "b.Motif de Consultation : Raisons du rendez-vous, symptômes rapportés."
                 "c.Compte-Rendu Médical : Examens cliniques, résultats d’imageries, bilans biologiques."
                 "d.Diagnostic : Résumé du diagnostic, stade de la maladie, score TNM si applicable."
                 "e.Traitements et Recommandations : Protocole de soins, traitements médicamenteux, décisions prises en réunion"
                 "pluridisciplinaire."
                 "f.Pronostic et Suivi : Recommandations à court et long terme, examens à programmer."
                 "g.Autres Notes Importantes : Informations pertinentes non catégorisables."
                 "4. Préciser l'anapathomologie et la biologie moléculaire (HER2, RH, RE)"
                 "5.Mettre en évidence les seuils cliniques, recommandations prioritaires et termes médicaux clés en utilisant de"
                 "puces ou des paragraphes explicites."
                 "6.Veiller à une structuration cohérente et uniforme pour garantir l’extraction complète et précise des données" 
                 "exploitables par un système de RAG."},
                {"type": "image_url", "image_url": f"data:image/png;base64,{image_base64}"}
            ]}
        ],
        max_tokens=15000
    )
    return response.choices[0].message.content


##################################
# Gère la fusion de listes de documents ou de messages
##################################
@component
class ListJoiner:
    def __init__(self, _type: Any):
        # Indiquer que la sortie est une liste
        component.set_output_types(self, values=List[_type])

    def run(self, values: Variadic[Any]):
        result = list(chain(*values))
        return {"values": result}

######################################
# Configuration de la page Streamlit
######################################
st.set_page_config(page_title="Sémiramis", layout="centered")

st.title("Sémiramis")
st.write(
    "Bienvenue sur Sémiramis, votre experte virtuelle en cancérologie sénologique.\n"
    "Chargez vos documents, posez vos questions, et recevez des réponses sur votre prise en charge."
)

######################################
# Récupération des clés depuis st.secrets
######################################
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    pinecone_api_key = st.secrets["PINECONE_API_KEY"]
except Exception:
    st.error("Veuillez configurer OPENAI_API_KEY et PINECONE_API_KEY dans .streamlit/secrets.toml.")
    st.stop()

if not openai_api_key or not pinecone_api_key:
    st.error("Veuillez configurer OPENAI_API_KEY et PINECONE_API_KEY dans .streamlit/secrets.toml.")
    st.stop()

# Définir les variables d'environnement (utilisées par Haystack)
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["PINECONE_API_KEY"] = pinecone_api_key


def create_new_patient_document_store():
    """Crée un nouveau document store Pinecone avec un ID compatible (minuscules, chiffres, tirets uniquement)."""
    
    # Générer un UUID et le rendre compatible avec Pinecone
    raw_uuid = str(uuid.uuid4())  # UUID original (ex: "550e8400-e29b-41d4-a716-446655440000")
    patient_id = re.sub(r'[^a-z0-9-]', '', raw_uuid)  # Garde uniquement a-z, 0-9, et '-'

    # Sauvegarder l'ID du patient dans la session
    st.session_state["patient_id"] = patient_id

    # Créer un nouvel index Pinecone
    patient_document_store = PineconeDocumentStore(
        api_key=Secret.from_env_var("PINECONE_API_KEY"),
        index=f"patient-{patient_id}",  # Utilisation d'un préfixe standardisé "patient-"
        dimension=3072
    )

    st.session_state["patient_document_store"] = patient_document_store
    #st.sidebar.success(f"Nouveau document store créé : patient-{patient_id}")

######################################
# Mémoire de conversation (Haystack + session_state)
######################################
if "memory_store" not in st.session_state or st.session_state.reset_chat:
    st.session_state.memory_store = InMemoryChatMessageStore()
    st.session_state.reset_chat = False

# Créer un nouveau document store au début de la session ou quand un nouveau patient est détecté
if "patient_document_store" not in st.session_state:
    create_new_patient_document_store()

##################################
# Pipeline Setup
##################################
if "rag_pipeline" not in st.session_state:
    knowledge_document_store = PineconeDocumentStore(
        api_key=Secret.from_env_var("PINECONE_API_KEY"),
        index="sein",
        dimension=3072
    )

    st.session_state["knowledge_document_store"] = knowledge_document_store

    # Création d'un document store unique pour le patient
    create_new_patient_document_store()

    # Système
    system_message = ChatMessage.from_system(
        "Vous êtes un oncologue médical spécialisé dans le cancer du sein. "
        "Vous répondez de manière concise aux questions en vous appuyant "
        "sur l'historique de conversation et les documents médicaux fournis. Si l'utilisateur recherche à comprendre," 
        "vous pouvez utiliser votre savoir pour l'éclairer."
        "Si la question ne peut être répondue à partir des documents, par manque d'informations', indiquez-le clairement et demandez des précisions."
        "Vous vous adressez à des patients ou des médecins dans le cadre de réunion de concertation pluri-disciplinaire" 
        "ou d'explications de résultats médicaux et de parcours personalisé de soins."
    )

    # Template utilisateur
    user_message_template = """
En vous basant en priorité sur l'historique et les documents ci-dessous, répondez de façon concise.

Historique:
{% for memory in memories %}
{{ memory.text }}
{% endfor %}

Documents:
{% for doc in documents %}
{{ doc.content }}
{% endfor %}

Si les documents ne suffisent pas éclairer l'utilisateur, utiliser votre savoir à des fins explicatives
Question: {{query}}
Réponse:
"""
    user_message = ChatMessage.from_user(user_message_template)

    # Prompt de reformulation
    query_rephrase_template = """
Réécrivez la question afin qu’elle soit adaptée à la recherche documentaire médicale,
sans en changer le sens ni les mots-clés médicaux essentiels. N'hésitez pas à demander
le statut des récepteurs hormonaux, le TNM et la taille de la tumeur si non fourni par l'utilisateur.

Exemple:
- Historique: ["Le patient veut plus de précisions"]
- Question originale: "Quels sont les risques de la chimiothérapie pour mon cancer ?"
- Question reformulée: "Risques de la chimiothérapie dans le traitement du cancer"

Historique:
{% for memory in memories %}
{{ memory.text }}
{% endfor %}

Question: {{query}}

Nouvelle question:

"""

    memory_store = st.session_state.memory_store
    memory_retriever = ChatMessageRetriever(memory_store)
    memory_writer = ChatMessageWriter(memory_store)

    # Pipeline
    conversational_rag = Pipeline()

    # Composants
    conversational_rag.add_component("memory_retriever", memory_retriever)
    conversational_rag.add_component("query_rephrase_prompt_builder", PromptBuilder(query_rephrase_template))
    conversational_rag.add_component("query_rephrase_llm", OpenAIGenerator(model="gpt-4o", generation_kwargs={"n": 1, "temperature": 0.21, "max_tokens": 7000}))
    conversational_rag.add_component("list_to_str_adapter", OutputAdapter(template="{{ replies[0] }}", output_type=str))

    # Embedding
    conversational_rag.add_component("query_embedding_generator", OpenAITextEmbedder(model="text-embedding-3-large"))

    # Retrievers
    # Vérification : si le document store patient n'existe pas, on en crée un nouveau
    if "patient_document_store" not in st.session_state:
      create_new_patient_document_store()

    retriever_patient = PineconeEmbeddingRetriever(document_store=st.session_state["patient_document_store"], top_k=3)
    retriever_knowledge = PineconeEmbeddingRetriever(document_store=knowledge_document_store, top_k=3)

    conversational_rag.add_component("retriever_patient", retriever_patient)
    conversational_rag.add_component("retriever_knowledge", retriever_knowledge)
    conversational_rag.add_component("documents_joiner", ListJoiner(Document))
    conversational_rag.add_component("hyde_generator", OpenAIGenerator(model="gpt-4o", generation_kwargs={"n": 1, "temperature": 0.21, "max_tokens": 7000}))
    conversational_rag.add_component("hyde_output_adapter", OutputAdapter(template="{{ replies[0] }}", output_type=str))



    # Prompt builder + LLM
    conversational_rag.add_component("prompt_builder", ChatPromptBuilder(variables=["query", "documents", "memories"], required_variables=["query", "documents", "memories"]))
    conversational_rag.add_component("llm", OpenAIChatGenerator(model="gpt-4o", generation_kwargs={"n": 1, "temperature": 0.21, "max_tokens": 7000}))

    # Memory writing
    conversational_rag.add_component("memory_writer", memory_writer)
    conversational_rag.add_component("memory_joiner", ListJoiner(ChatMessage))

    # Connections
    # Rephrase
    conversational_rag.connect("memory_retriever", "query_rephrase_prompt_builder.memories")
    conversational_rag.connect("query_rephrase_prompt_builder.prompt", "query_rephrase_llm")
    conversational_rag.connect("query_rephrase_llm.replies", "list_to_str_adapter")
    #conversational_rag.connect("list_to_str_adapter", "query_embedding_generator.text")

    # Générer un passage hypothétique à partir de la requête reformulée
    conversational_rag.connect("list_to_str_adapter", "hyde_generator.prompt")

    # Adapter la sortie du HyDE pour qu'elle soit une chaîne unique
    conversational_rag.connect("hyde_generator.replies", "hyde_output_adapter")

    # Envoyer la version enrichie à l'embedding
    conversational_rag.connect("hyde_output_adapter", "query_embedding_generator.text")


    # Retrieve
    conversational_rag.connect("query_embedding_generator.embedding", "retriever_patient.query_embedding")
    conversational_rag.connect("query_embedding_generator.embedding", "retriever_knowledge.query_embedding")
    conversational_rag.connect("retriever_patient.documents", "documents_joiner.values")
    conversational_rag.connect("retriever_knowledge.documents", "documents_joiner.values")
    conversational_rag.connect("documents_joiner", "prompt_builder.documents")

    # LLM
    conversational_rag.connect("prompt_builder.prompt", "llm.messages")

    # Memory
    conversational_rag.connect("llm.replies", "memory_joiner")
    conversational_rag.connect("memory_joiner", "memory_writer")
    conversational_rag.connect("memory_retriever", "prompt_builder.memories")

    st.session_state.rag_pipeline = conversational_rag
    st.session_state.system_message = system_message
    st.session_state.user_message_template = user_message
    
##################################
# 6) Upload et indexation des PDF (OCR GPT)
##################################
st.sidebar.header("Téléversement de vos documents (compte rendu, analyse biologiques, etc.)")

uploaded_file = st.sidebar.file_uploader("Glisser vos fichiers ci-dessous", type=["pdf"])

# Bouton de réinitialisation en bas de la sidebar
st.sidebar.markdown("---")
reset_button = st.sidebar.button("Réinitialiser la conversation")
if reset_button:
    st.session_state.chat_history = []
    st.session_state.memory_store = InMemoryChatMessageStore()
    st.session_state.reset_chat = True
    st.sidebar.success("Conversation réinitialisée!")


if uploaded_file is not None:
    # 1) Sauvegarde temporaire du PDF
    pdf_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 2) Extraction en images
    output_folder = "temp/images"
    image_paths = extract_images_from_pdf(pdf_path, output_folder)

    # 3) Appel OCR GPT page par page
    st.write("Extraction et structuration de vos documents...")
    extracted_texts = []
    for img_path in image_paths:
        page_text = process_image_with_openai(img_path)
        extracted_texts.append(page_text)

    # Fusion de tout le texte
    full_text = "\n".join(extracted_texts)

    # 4) Indexation dans le store Pinecone "patient"
    #    => On va créer un mini-pipeline pour convertir puis indexer
    indexing_patient = Pipeline()
    indexing_patient.add_component("converter", TextFileToDocument())
    indexing_patient.add_component("cleaner", DocumentCleaner())
    indexing_patient.add_component("splitter", DocumentSplitter(split_by="sentence", split_length=5))
    indexing_patient.add_component("embedder", OpenAIDocumentEmbedder(model="text-embedding-3-large"))
    indexing_patient.add_component("writer", DocumentWriter(st.session_state.patient_document_store))

    indexing_patient.connect("converter", "cleaner")
    indexing_patient.connect("cleaner", "splitter")
    indexing_patient.connect("splitter", "embedder")
    indexing_patient.connect("embedder", "writer")


    # Pour indexer, on va sauvegarder `full_text` dans un fichier temporaire .txt
    extracted_text_file = os.path.join("temp", "extracted_text.txt")
    with open(extracted_text_file, "w", encoding="utf-8") as tmpf:
        tmpf.write(full_text)

    # On lance l’indexation
    indexing_patient.run({
        "converter": {
            "sources": [extracted_text_file],
            "meta": {"doc_type": "patient_doc"}
        }
    })
    st.sidebar.success("Documents analysés et enregistrés !")    


######################################
# Gestion de l'historique « en direct »
######################################
if "chat_history" not in st.session_state or st.session_state.reset_chat:
    st.session_state.chat_history = []

# On affiche ici tous les messages existants de la conversation.
for role, content in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(content)

# On récupère la saisie utilisateur via la zone de chat.
user_query = st.chat_input("Laissez moi vous aidez avec vos documents et votre parcours de soins...")

# Lorsqu'il y a un nouveau message :
if user_query:
    # 1) On affiche tout de suite le message utilisateur.
    st.session_state.chat_history.append(("user", user_query))
    with st.chat_message("user"):
        st.write(user_query)

    # 2) On exécute le pipeline Haystack pour générer la réponse.
    pipeline = st.session_state.rag_pipeline
    system_msg = st.session_state.system_message
    user_msg = st.session_state.user_message_template

    result = pipeline.run(
        data={
            "query_rephrase_prompt_builder": {"query": user_query},
            "prompt_builder": {
                "template": [system_msg, user_msg],
                "query": user_query
            },
            "memory_joiner": {"values": [ChatMessage.from_user(user_query)]}
        },
        include_outputs_from=["llm", "query_rephrase_llm"]
    )

    assistant_resp = result["llm"]["replies"][0].text

    # 3) On affiche immédiatement la réponse de l’assistant.
    st.session_state.chat_history.append(("assistant", assistant_resp))
    with st.chat_message("assistant"):
        st.write(assistant_resp)