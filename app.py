from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import textwrap
from urllib.parse import urlparse, parse_qs
from langchain.tools import YouTubeSearchTool
import streamlit as st
import re
from dotenv import load_dotenv
from pytube import YouTube

# Load environment variables from .env file
load_dotenv()

embeddings = OpenAIEmbeddings()

searcher = YouTubeSearchTool()

def get_choice():
    """Get the user's choice for search method."""
    print("How would you like to search for videos?")
    print("1. By YouTube URL")
    print("2. By youtuber's name or topic")
    choice = input("Enter your choice (1 or 2): ")
    return choice


def get_video_id_from_url(video_url):
    """Extract the video ID from a YouTube URL."""
    query = urlparse(video_url).query
    video_id = parse_qs(query).get("v", [None])[0]
    return video_id if video_id else None


def create_db_from_youtube_video_url(video_url):
    video_id = get_video_id_from_url(video_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL. Unable to extract video ID.")

    loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=True)
    transcript = loader.load()

    # Using pytube to get the video title
    yt = YouTube(video_url)
    video_title = yt.title

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)

    return db, video_title



def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    chat = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.2)

    # Template to use for the system message prompt
    template = """
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs


def main():
    st.title("YouTube Video Summarizer")

    choice = st.radio("How would you like to search for videos?", 
                  ("By YouTube URL(s)", "By youtuber's name or topic"), 
                  index=1)  # Set the second option as default
    
        # Choose the type of AI instruction
    instruction_choice = st.radio("Choose the type of instruction for the AI:", 
                                  ("Default Summary", "Custom Instruction"), 
                                  index=0)

    custom_instruction = ""
    if instruction_choice == "Custom Instruction":
        custom_instruction = st.text_area("Write your custom instruction for the AI:")


    video_urls = []

    if choice == "By YouTube URL(s)":
        raw_video_urls = st.text_area("Please enter the YouTube URLs (one per line or separated by commas):")
        
        if st.button("Process URLs"):
            # Split the input by newline or comma
            split_urls = [url.strip() for url in re.split('[,\n]', raw_video_urls) if url.strip()]
            for video_url in split_urls:
                if "youtube.com" not in video_url:
                    st.error(f"Invalid YouTube URL: {video_url}")
                    return
                else:
                    video_urls.append(video_url)

    elif choice == "By youtuber's name or topic":
        items = st.number_input("How many videos do you want to search?", min_value=1, step=1, value=5)
        search_item = st.text_input("What is the name of the YouTuber or topic?")
        if st.button("Search"):
            try:
                video_suffixes = searcher.run(f'{search_item},{items}')
                video_suffixes_cleaned = video_suffixes.replace("[", "").replace("]", "").replace("'", "")
                # Ensure that the base URL is not duplicated
                video_urls.extend([suffix.strip() for suffix in video_suffixes_cleaned.split(',')])
            except ValueError:
                st.error("Error occurred during search. Please try again.")
    if video_urls:
        st.write("Processing the following URLs:", video_urls)

        summaries = []  # List to store summaries for each video
        for video_url in video_urls:
            try:
                db, video_title = create_db_from_youtube_video_url(video_url)
                # Determine the query based on user choice
                if instruction_choice == "Default Summary":
                    query = """
    Given the content from the YouTube video, please provide a concise summary highlighting the main points and important details. 
    Organize the information with bullet points for clarity.
    Do not mention about: The speaker mentions a link in the description for more information. OR The video encourages viewers to like and subscribe to the channel.
    """
                else:
                    query = custom_instruction  # Use the custom instruction provided by the user

                response, docs = get_response_from_query(db, query)
                summaries.append((video_url, video_title, textwrap.fill(response, width=50)))
            except Exception as e:
                st.error(f"Error processing video {video_url}. Error: {e}")

        # Display the summaries
        for url, title, summary in summaries:
            st.subheader(f"Video Title: {title}")
            st.markdown(f"Video URL: {url}")
            st.write(f"Summary: {summary}")
            st.write('-' * 50)


if __name__ == "__main__":
    main()