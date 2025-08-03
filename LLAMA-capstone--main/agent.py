
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
import requests
from transformers.pipelines.text_generation import TextGenerationPipeline


url = "https://huggingface.co/datasets/WahajSa/Productsinfo/resolve/main/KB_Products.txt"
response = requests.get(url)
text = response.text


docs = [Document(page_content=text)]
text_splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=100)
documents = text_splitter.split_documents(docs)


embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")



vectorstore = Chroma.from_documents(
    documents,
    embedding,
    persist_directory="chromadb"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "Yasser18/aps3010-finetuned-model"


# model = AutoModelForCausalLM.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# pipe = TextGenerationPipeline(
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=500,
#     temperature=0.7
# )
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key="AIzaSyDxcmxnwx9cj5G7hnnLDg63XmwbubzAymo",
    temperature=0.7
)

# llm = HuggingFacePipeline(pipeline = pipe)

from langchain.tools.retriever import create_retriever_tool

# p1: APS3010H power supply tool
p2_tool = create_retriever_tool(
    retriever,
    name="T12PRO",
   description="Use this tool to answer any question about the T12 iron soldering, including who the supervisors or developers are, features, or people or the model involved in its creation.",
)

p1_tool = create_retriever_tool(
     retriever,
     name="APS3010H",
    description="Use this tool to answer any question about the APS3010H dc powersupply, including who the supervisors or developers are, features, or people involved in its creation.",
 )



from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant specialized in mechanical and electronic products.
All questions refer to the device or product model described in the context — never to yourself as an AI model.
When asked about a model, it means a product model, not you.


{context}

Question: {question}
Answer:
"""
)

from langchain.agents import initialize_agent
from langchain.agents import AgentType

agent_executor = initialize_agent(
    tools=[p1_tool,p2_tool],
    llm=llm,
    memory= memory,
    prompt = prompt,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True
)

