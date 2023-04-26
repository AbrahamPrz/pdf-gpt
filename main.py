# Code obtained from https://www.youtube.com/watch?v=TLf90ipMzfE
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
MATH_STANDARDS = os.path.join(AZ_STANDARDS, 'mathematics')
SCIENCE_STANDARDS = os.path.join(AZ_STANDARDS, 'science')


def main(grade: str, subject: str) -> None:
    pdf_texts = []
    raw_text = ''
    texts = []
    
    # get pdf files based on subject and grade
    if subject == '1':
        pdf_files = [os.path.join(MATH_STANDARDS, f) for f in os.listdir(MATH_STANDARDS) if f.lower().endswith('.pdf')]
    else:
        if grade == '1':
            pdf_files = [os.path.join(SCIENCE_STANDARDS, '1_FirstGradeScienceStandardsPlacemat.pdf')]
        elif grade == '2':
            pdf_files = [os.path.join(SCIENCE_STANDARDS, '2_SecondGradeScienceStandardsPlacemat.pdf')]
        else:
            pdf_files = [os.path.join(SCIENCE_STANDARDS, '3_ThirdGradeScienceStandardsPlacement.pdf')]
    
    # convert pdf files to text
    for pdf in pdf_files:
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
    
    # ask questions
    query = input("Ask a question: ")
    
    while query.lower() != 'quit':
        docs = docsearch.similarity_search(query)
        answer = chain.run(input_documents=docs, question=query)
        
        print(f'Answer: {answer}')    
        query = input("Please enter a question: ")


if __name__ == '__main__':
    grade = input("Select a grade: \n1. First Grade\n2. Second Grade\n3. Third Grade\n> ")
    while grade not in ['1', '2', '3']:
        grade = input("Please select one of the three options listed above:\n> ")
    
    subject = input("Select a subject: \n1. Math\n2. Science\n> ")
    while subject not in ['1', '2']:
        subject = input("Please select the number of one of the options listed above:\n> ")
    
    main(grade=grade, subject=subject)
