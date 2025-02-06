import streamlit as st
import pdfplumber
from langchain.schema import Document
from langchain import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


## sstreamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT, Website or PDF", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT, Website or a PDF")
st.subheader('QuickSum: Effortless Summaries for a Smarter Life')


## Get the Groq API Key and url(YT or website)to be summarized
with st.sidebar:
    #GROQ API
    groq_api_key = st.text_input("Groq API Key",value= "", type="password")  # Do not hard-code API key
    # Type of documnet to be summarised
    option = st.selectbox("What is the source of the document", ("You Tube", "Website URL", "PDF"))
    st.write("You selected:", option)
    # Model to be selected from GRQO Cloud
    model = st.selectbox("Select a model for Summarisation", ("deepseek-r1-distill-llama-70b", "gemma2-9b-it", "llama-3.3-70b-versatile", "whisper-large-v3-turbo","mixtral-8x7b-32768"))
    # User definded prompt for summarization task
    user_prompt = st.text_area("How do you envision the summary?",height=350)
    st.write(f"You wrote {len(user_prompt)} characters.")

  
#Inputs for summarisation
st.write('Enter you URL here:')
generic_url=st.text_input("URL",label_visibility="collapsed")

st.write("or")


#Importing the uploaded pdf file
data = []

def extract_data(feed):
    with pdfplumber.open(uploaded_file) as pdf:
        pages = pdf.pages
        for p in pages:
            data.append(p.extract_text())
    return data

uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")
if uploaded_file is not None:
    df = extract_data(uploaded_file)


#Custom Prompt design for the LLM Model
prompt_template="""
Provide a summary of the following content in 300 words:
Content:{text}

"""
prompt=PromptTemplate(template=prompt_template,input_variables=["text"])

if st.button("Summarize the Content"):
    ## Validate all the inputs
    if not groq_api_key.strip():
        st.error("Please provide the API Key")
        

    else:
        ## Different Models that can be used using Groq API
        if model =="deepseek-r1-distill-llama-70b":
            llm =ChatGroq(model="deepseek-r1-distill-llama-70b", groq_api_key=groq_api_key)
        elif model=="":
            llm =ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)
        elif model=="":
            llm =ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)
        elif model=="":
            llm =ChatGroq(model="whisper-large-v3-turbo", groq_api_key=groq_api_key)
        else:
            llm =ChatGroq(model="mixtral-8x7b-32768", groq_api_key=groq_api_key)

        try:
            with st.spinner("Waiting..."):
                ## loading the website or yt video data
                if option == "You Tube":
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=False)  # Set add_video_info to True
                    docs = loader.load()
                
                elif option == "Website URL":
                    loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                 headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                    docs = loader.load()
                
                else:
                    docs = [Document(page_content=text) for text in df]
                    

                instruction = user_prompt
                # Splitting the document into chunks
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                final_documents=splitter.split_documents(docs)
                # Prompt for map
                chunks_prompt="""
                                You are an intelligent assistant tasked with summarizing a given piece of text. Your job is to condense the provided text into a shorter version that highlights the most important information while preserving the meaning and key points. The summary should be clear, concise, and contain only the essential details, avoiding any unnecessary elaboration or irrelevant information.

                                Please read the following passage carefully, and then generate a summary that includes the following:
                                1. The key concepts or events discussed in the text.
                                2. The most important details that provide context or explanation for the main points.
                                3. Any conclusions or noteworthy takeaways.

                                Do not add any extra information that is not present in the original text. Your summary should be in full sentences and should be easy to understand. 

                                Here is the text to summarize:

                                "{text}"

                                Please provide your summarized version of the text below:

                            """
                map_prompt_template=PromptTemplate(input_variables=['text'],template=chunks_prompt)

                # User defined prompt for final summarisation in reduce stage
                final_prompt='''
                Provide the final summary :{instruction}
                Speech:{text}

                '''
                final_prompt_template=PromptTemplate(input_variables=['instruction','text'],template=final_prompt)
                

                ## Chain For Summarization
                summary_chain=load_summarize_chain(
                                llm=llm,
                                chain_type="map_reduce",
                                map_prompt=map_prompt_template,
                                combine_prompt=final_prompt_template,
                                verbose=True)
                
                output_summary=summary_chain.run({'instruction':instruction,'input_documents': docs})

                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception:{e}")
