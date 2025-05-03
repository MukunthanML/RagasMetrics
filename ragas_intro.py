"""
This script demonstrates the usage of the `ragas` library to evaluate a single-turn
question-answering (QA) sample using the `StringPresence` metric.
Modules:
    - `SingleTurnSample`: A class from the `ragas` library used to represent a single-turn QA sample.
    - `StringPresence`: A metric from the `ragas.metrics` module used to evaluate the presence of a 
      reference string in the response.
Test Data:
    - `user_input`: The question asked by the user.
    - `response`: The answer provided by the system.
    - `reference`: The expected reference string to be checked in the response.
Workflow:
    1. Define a dictionary `test_data` containing the user input, system response, and reference string.
    2. Instantiate the `StringPresence` metric.
    3. Convert the `test_data` dictionary into a `SingleTurnSample` object.
    4. Evaluate the response using the `single_turn_score` method of the `StringPresence` metric.
    5. Print the resulting score.
Usage:
    This script can be used to evaluate the quality of a single-turn QA system's response by checking
    if the reference string is present in the response.
"""
from ragas.metrics import StringPresence
from ragas import SingleTurnSample


test_data = {
    "user_input":"What is the capital of Germany?",
    "response":"The capital of Germany is Berlin.",
    "reference":"Berli",
    }
    
metric = StringPresence()
test_data = SingleTurnSample(**test_data)
print(metric.single_turn_score(test_data))

