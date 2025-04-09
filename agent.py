from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model= OllamaLLM(model="gemma3:1b")

template = """
 You are an expert in answering questions about a pizza restaurant

 Here are some reviews: {reviews}

 Here is the questions to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model 

while True:
    question= input("Ask your question (q to quit):")
    if question == "q":
        break

    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews": reviews, "question" : question})
    print(result)
