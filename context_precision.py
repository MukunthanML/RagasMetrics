import asyncio
from ragas.dataset_schema import SingleTurnSample  
from ragas.metrics import LLMContextPrecisionWithoutReference, LLMContextPrecisionWithReference
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


sample_without_ref = SingleTurnSample(
    user_input="When was the first super bowl?",  # Question asked by the user
    response="First AFL–NFL World Championship was at Los Angeles.", 
    retrieved_contexts=[
        "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles.",
        "The game was played between the American Football League (AFL) champion Kansas City Chiefs and the National Football League (NFL) champion Green Bay Packers.",
        "The Packers won the game by a score of 35 to 10.",
        "The game was the first championship game of the NFL and is now known as Super Bowl I.",
        "The game was played on January 15, 1967, and was the first Super Bowl in history.",
        "The game was played at the Los Angeles Memorial Coliseum in Los Angeles, California.",
        "The game was attended by 61,946 fans and was broadcast on television in the United States by NBC and CBS.",
        "The game was also broadcast on radio by the Mutual Broadcasting System.",
        "The game was played in front of a crowd of 61,946 fans at the Los Angeles Memorial Coliseum.",
    ]
)

sample_with_ref = SingleTurnSample(
     user_input="Where is the Eiffel Tower located?",
    # reference="The Eiffel Tower is located in Paris.",
    retrieved_contexts=["Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars.","It is in Paris, France.",
                        "The tower was completed in 1889 as the entrance arch to the 1889 World's Fair.",
                        ], 
)

# Initialize the Response Relevancy scorer with the wrapped LLM
context_precision_without_ref= LLMContextPrecisionWithoutReference(llm=evaluator_llm)
context_precision_with_ref= LLMContextPrecisionWithReference(llm=evaluator_llm)


# Define the main asynchronous function to calculate the faithfulness score
async def main():
    # Calculate the faithfulness score for the given sample
    context_precision_without_ref_score = await context_precision_without_ref.single_turn_ascore(sample_without_ref)
    context_precision_with_ref_score = await context_precision_with_ref.single_turn_ascore(sample_with_ref)
    # Print the calculated faithfulness score
    print(f"Context Precision (without reference) : {context_precision_without_ref_score}")
    print(f"Context Precision (with reference) : {context_precision_with_ref_score}")

# Run the main asynchronous function
asyncio.run(main())

