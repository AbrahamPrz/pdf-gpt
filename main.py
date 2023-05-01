import gradio as gr
import time
import os
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from dotenv import load_dotenv


load_dotenv()
# Make sure to set your OpenAI API key as an environment variable:
# OPENAI_API_KEY = 'your-api-key-here'


BASE_DIR = os.path.dirname(os.path.realpath(__file__))
AZ_STANDARDS = os.path.join(BASE_DIR, 'az-standards')
SUBJECT_LIST = ["Math", "Science"]
GRADE_LIST = [1, 2, 3]



with gr.Blocks() as demo:
    
    subject = gr.Radio(choices=SUBJECT_LIST, label="Subject", value=SUBJECT_LIST[0], interactive=True)
    grade = gr.Radio(choices=GRADE_LIST, label="Grade", value=GRADE_LIST[0], interactive=True)
    
    msg = gr.Textbox(placeholder="Type your message here")
    chatbot = gr.Chatbot()
    clear = gr.Button("Clear")
    
    def user(user_message, history):
        return user_message, history + [[user_message, None]]

    def bot(subject, grade, user_message, history):
        pdf_paths = []
        
        pdf_paths.append(os.path.join(AZ_STANDARDS, 'base', 'lesson_plan_template.pdf'))
        # The template provided allows the LM to understand the context of the question and provide an example for a lesson plan.
        
        subject_selection, grade_selection = SUBJECT_LIST.index(subject), GRADE_LIST.index(grade)
        
        az_standards_dir = os.listdir(AZ_STANDARDS)
        az_standards_dir.sort()
            
        subject_path = az_standards_dir[subject_selection+1]
        full_subject_path = os.path.join(AZ_STANDARDS, subject_path)
        
        subject_dir = os.listdir(full_subject_path)
        subject_dir.sort()
        
        grade_path = subject_dir[grade_selection+1]
        full_grade_path = os.path.join(full_subject_path, grade_path)
        
        downloadable_files = []
        
        for pdf in os.listdir(full_grade_path):
            pdf_paths.append(os.path.join(full_grade_path, pdf))
        
        pdf_texts = []
        raw_text = ''
        texts = []
        
        # convert pdf files to text
        for pdf in pdf_paths:
            downloadable_files.append(gr.File(value=pdf, label="Context files"))
            try :
                reader = PdfReader(pdf)
            except:
                pass
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    raw_text += text
            pdf_texts.append(raw_text)
            raw_text = ''

        # split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        for pdf_text in pdf_texts:
            texts += text_splitter.split_text(pdf_text)
        
        # create embeddings
        embeddings = OpenAIEmbeddings()
        docsearch = FAISS.from_texts(texts, embeddings)
        chain = load_qa_chain(OpenAI(model_name='gpt-3.5-turbo', temperature=0.9), chain_type="stuff")
        
        docs = docsearch.similarity_search(user_message)
        bot_message = chain.run(input_documents=docs, question=user_message)
        
        history[-1][1] = ""
        for character in bot_message:
            history[-1][1] += character
            # time.sleep(0.05)
            yield '', history
    
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, [subject, grade, msg, chatbot], [msg, chatbot]
    )
    clear.click(lambda: None, None, chatbot, queue=False)


demo.queue()
demo.launch(share=True)
