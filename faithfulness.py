import asyncio
from ragas.dataset_schema import SingleTurnSample  
from ragas.metrics import Faithfulness
from ragas.llms import LangchainLLMWrapper 
from langchain_groq import ChatGroq 

# Initialize the ChatGroq LLM with specific parameters
llm = ChatGroq(
    model="deepseek-r1-distill-llama-70b", 
    temperature=0.0,  # Temperature for deterministic responses
    max_retries=2  # Maximum retries for API calls
    # set GROQ_API_KEY to your environment variable or uncomment the line below to set it directly
    # groq_api_key="your_groq_api_key"  # Uncomment and set your API key if needed
)

# Wrap the LLM using LangchainLLMWrapper for compatibility with Ragas
evaluator_llm = LangchainLLMWrapper(llm)

# Define a single-turn QA sample with user input, response, and retrieved contexts
sample = SingleTurnSample(
    user_input="When was the first super bowl?",  # Question asked by the user
    response="First AFL–NFL World Championship was an American football.",  # faithful answer, but not relevant
    # response="The first superbowl was held on Jan 15, 1967",  # faithful answer and relevants
    retrieved_contexts=[
        # Context retrieved to support the response
        "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles."
    ]
)

# Initialize the Faithfulness scorer with the wrapped LLM
faithfulness_scorer = Faithfulness(llm=evaluator_llm)

# Define the main asynchronous function to calculate the faithfulness score
async def main():
    # Calculate the faithfulness score for the given sample
    faith_score = await faithfulness_scorer.single_turn_ascore(sample)
    # Print the calculated faithfulness score
    print(f"User Input : {sample.user_input}")
    print(f"Original LLM Response : {sample.response}")
    print(f"Retrieved context : {sample.retrieved_contexts}")
    print(f"Faithfulness Score (by evaluator LLM) : {faith_score}")

# Run the main asynchronous function
asyncio.run(main())

