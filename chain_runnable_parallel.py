from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel

load_dotenv()
model = ChatOpenAI(model="gpt-5")

# Defining Prompt template for Movie Summary
summary_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a movie critic."),
        ("human","Provide a brief summary of the movie {movie_name}.")
    ]
)

# Defining plot analysis step
def analyze_plot(plot):
    plot_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a movie critic."),
            ("human","Analyze the plot: {plot}. What are its strengths and weakness?")
        ]
    )

    return plot_template.format_prompt(plot=plot)

# Defining characters analysis step
def analyze_characters(characters):
    character_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a movie critic."),
            ("human","Analyze the characters: {characters}. What are its strengths and weakness?")
        ]
    )
    return character_template.format_prompt(characters=characters)


def combine_verdict(plot_analysis, character_analysis):

    return f"Plot Analyis:\n {plot_analysis} \n\n Character Analysis:\n {character_analysis}"


plot_branch_chain = (
    RunnableLambda(lambda x: analyze_plot(x)) | model | StrOutputParser()
)

character_branch_chain = (
    RunnableLambda(lambda x: analyze_characters(x)) | model | StrOutputParser()
)

chain = (
    summary_template
    | model
    | StrOutputParser()
    | RunnableParallel(branches={"plot": plot_branch_chain, "characters":character_branch_chain})
    | RunnableLambda(lambda x: combine_verdict(x["branches"]["plot"], x["branches"]["characters"]))
)

result = chain.invoke({"movie_name":"Akhanda"})

print(result)

