import streamlit as st
import pdfplumber
from langchain.schema import Document
from langchain import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from gtts import gTTS
from io import BytesIO


# Model initialisation for text summarisation
def initialize_groq_model(model_name):
    """Initialize Groq model based on user selection."""
    try:
        model_map = {
            "deepseek-r1-distill-llama-70b": "deepseek-r1-distill-llama-70b",
            "gemma2-9b-it": "gemma2-9b-it",
            "llama-3.3-70b-versatile": "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768": "mixtral-8x7b-32768"
        }
        model = model_map.get(model_name)
        if not model:
            raise ValueError(f"Model {model_name} is not recognized.")
        st.write(f"Model {model_name} initialized successfully.")
        return ChatGroq(model=model, groq_api_key=groq_api_key) 
    except Exception as e:
        st.error(f"Error initializing Groq model: {e}")
        raise


def extract_pdf_data(uploaded_file):
    """Extract text from uploaded PDF."""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            return [page.extract_text() for page in pdf.pages if page.extract_text()]
    except Exception as e:
        st.error(f"Error extracting data from PDF: {e}")
        return []


def summarize_document(docs, user_prompt, model):
    """Summarize the document using LangChain."""
    try:
        if not docs:
            st.error("No documents to summarize.")
            return ""

        # Split the document into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        final_documents = splitter.split_documents(docs)

        # Create prompt templates
        chunks_prompt = """
            You are an intelligent assistant tasked with summarizing a given piece of text. Your job is to condense the provided text into a shorter version that highlights the most important information while preserving the meaning and key points. The summary should be clear, concise, and contain only the essential details, avoiding any unnecessary elaboration or irrelevant information.

            Here is the text to summarize:
            "{text}"

            Please provide your summarized version of the text below:
        """
        map_prompt_template = PromptTemplate(input_variables=['text'], template=chunks_prompt)

        final_prompt = """
            Provide the final summary in the following language: {language}
            Follow the instruction given by the use to create final summary: {instruction}
            Speech: {text}
        """
        final_prompt_template = PromptTemplate(input_variables=['language','instruction', 'text'], template=final_prompt)

        # Chain for summarization
        summary_chain = load_summarize_chain(
            llm=model,
            chain_type="map_reduce",
            map_prompt=map_prompt_template,
            combine_prompt=final_prompt_template,
            verbose=True
        )

        output_summary = summary_chain.run({'language':language,'instruction': user_prompt, 'input_documents': docs})
        st.write("Summarization complete.")
        return output_summary
    except Exception as e:
        st.error(f"Error during summarization process: {e}")
        return ""

# Language for audio 
ietf_language_tags = {
    'English': 'en',
    'Spanish': 'es',
    'French': 'fr',
    'German': 'de',
    'Italian': 'it',
    'Portuguese': 'pt',
    'Dutch': 'nl',
    'Russian': 'ru',
    'Chinese': 'zh',
    'Japanese': 'ja',
    'Korean': 'ko',
    'Arabic': 'ar',
    'Hindi': 'hi',
    'Turkish': 'tr'
}

# Streamlit app setup
st.set_page_config(page_title="LangChain: Summarize Text From YT, Website or PDF", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT, Website or a PDF")
st.subheader('QuickSum: Effortless Summaries for a Smarter Life')

# Sidebar inputs
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")#Edit this portion of the code to input
    option = st.selectbox("What is the source of the document", ("PDF", "Website URL", "YouTube"))
    st.write("You selected:", option)
    language = st.selectbox('Please select the Language',(key for key,value in ietf_language_tags.items()))
    model_name = st.selectbox("Select a model for Summarization", (
        "deepseek-r1-distill-llama-70b", "gemma2-9b-it", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"))
    user_prompt = st.text_area("How do you envision the summary?", height=350)
    st.write(f"You wrote {len(user_prompt)} characters.")

# URL or PDF input
generic_url = st.text_input('Enter your URL here:')
uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")
data = []

if uploaded_file:
    data = extract_pdf_data(uploaded_file)

if st.button("Summarize the Content"):
    try:
        if not groq_api_key.strip():
            st.error("Please provide the API Key")


        # Initialize the selected model
        model = initialize_groq_model(model_name)

        # Load document based on selected option (YouTube, Website, PDF)
        docs = []
        if option == "YouTube":
            if generic_url:
                loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=False)
                docs = loader.load()
            else:
                st.error("Please provide a YouTube URL.")
    
        elif option == "Website URL":
            if generic_url:
                loader = UnstructuredURLLoader(urls=[generic_url], ssl_verify=False,
                                                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                docs = loader.load()
            else:
                st.error("Please provide a Website URL.")

        elif option == "PDF" and uploaded_file:
            docs = [Document(page_content=text) for text in data]
        else:
            st.error("Please provide a valid URL or PDF file.")


        # Summarize the document
        output_summary = summarize_document(docs, user_prompt, model)
        if output_summary:
            st.success(output_summary)
            sound_file = BytesIO()
            tts = gTTS(output_summary, lang=ietf_language_tags[language])
            tts.write_to_fp(sound_file)
            st.audio(sound_file)
        else:
            st.error("Error: No summary generated.")
    except Exception as e:
        st.exception(f"An unexpected error occurred: {e}")
