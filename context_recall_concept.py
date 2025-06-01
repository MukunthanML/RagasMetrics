# Hypothetical ground truth relevant documents (IDs)
ground_truth_relevant_ids = {1, 3, 5, 7}

# Hypothetically retrieved document IDs
retrieved_ids = {1, 4, 5, 8}

# Calculate the number of relevant retrieved documents
relevant_retrieved = len(ground_truth_relevant_ids.intersection(retrieved_ids))

# Calculate Context Recall
context_recall = relevant_retrieved / len(ground_truth_relevant_ids)

print(f"Ground Truth Relevant IDs: {ground_truth_relevant_ids}")
print(f"Retrieved IDs: {retrieved_ids}")
print(f"Relevant Retrieved Documents: {relevant_retrieved}")
print(f"Context Recall: {context_recall:.2f}")