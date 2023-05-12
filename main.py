import gradio as gr
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
    pdf_paths = []
    
    subject = gr.Radio(choices=SUBJECT_LIST, label="Subject", value=SUBJECT_LIST[0], interactive=True)
    grade = gr.Radio(choices=GRADE_LIST, label="Grade", value=GRADE_LIST[0], interactive=True)
    
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Type your message here")
    clear = gr.Button("Clear")
    files_used = gr.Files(value=[], label="Files used")
    
    
    def update_list(subject, grade, files_used):
        pdf_paths.clear()
        subject_selection, grade_selection = SUBJECT_LIST.index(subject), GRADE_LIST.index(grade)
        
        az_standards_dir = os.listdir(AZ_STANDARDS)
        az_standards_dir.sort()
            
        subject_path = az_standards_dir[subject_selection+1]
        full_subject_path = os.path.join(AZ_STANDARDS, subject_path)
        
        subject_dir = os.listdir(full_subject_path)
        subject_dir.sort()
        
        grade_path = subject_dir[grade_selection+1]
        full_grade_path = os.path.join(full_subject_path, grade_path)

        for pdf in os.listdir(full_grade_path):
            pdf_paths.append(os.path.join(full_grade_path, pdf))
                
        # The template provided allows the LM to understand the context of the question and provide an example for a lesson plan.
        pdf_paths.append(os.path.join(AZ_STANDARDS, 'base', 'lesson_plan_template.pdf'))
        
        files_used = pdf_paths
        return files_used
    
        
    def user(user_message, history):
        return user_message, history + [[user_message, None]]


    def bot(user_message, history, subject, grade):
        pdf_texts = []
        raw_text = ''
        texts = []
        
        # convert pdf files to text
        for pdf in pdf_paths:
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
        chain = load_qa_chain(OpenAI(model_name='gpt-3.5-turbo', temperature=0.4), chain_type="stuff")
        
        docs = docsearch.similarity_search(user_message, include_metadata=True)
        bot_message = chain.run(
            input_documents=docs, 
            question=f'''
                You're a friendly chatbot. \
                You can reply to basic interactions, like greetings, but you can only answer questions about the Arizona State Standards for the subject {subject} and grade {grade}.
                
                Answer the following text delimited by triple backticks only in the given context ```{user_message}```.
                
                Use this chat history to help you answer the question, delimited by ***, if needed. \
                The format of this chat history is a list of lists, where the first element of each list is the user message, the second element is the bot message \
                and the newer list is the latest chat interaction. \
                ***{history}***
                
                If you're asked to provide an example, please provide an example. \
                If you're asked to do a lesson plan, please provide a lesson plan.
                
                IMPORTANT: If you aren't provided with an example or a lesson plan petition, don't reply saying that you weren't provided with one. \
                If the user ask for things not related to the Arizona State Standards, please reply kindly that you can only answer Arizona State Standards related questions.
                '''
        ) if user_message else "I can't understand you."
        
        # print(str(docs)) # We can use this to see the metadata of the documents that were used to answer the question!.
        
        history[-1][1] = ""
        for character in bot_message:
            history[-1][1] += character
            # time.sleep(0.05)
            yield '', history

    subject.change(update_list, [subject, grade, files_used], files_used)
    subject.change(lambda: None, None, chatbot, queue=False)
    grade.change(update_list, [subject, grade, files_used], files_used)
    grade.change(lambda: None, None, chatbot, queue=False)

    msg.submit(fn=user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False).then(
        fn=bot, inputs=[msg, chatbot, subject, grade], outputs=[msg, chatbot]
    )
    clear.click(lambda: None, None, chatbot, queue=False)

    # Runs the function at the beginning to populate the list of files used.
    demo.load(update_list, [subject, grade, files_used], files_used)

demo.queue()
demo.launch(share=True)
