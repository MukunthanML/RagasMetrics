### 📌 **Context Precision (with and without reference) – Explained Simply**

In **LLM evaluation**, especially in **RAG (Retrieval-Augmented Generation)** systems, **Context Precision** is a metric used to measure **how relevant the retrieved context is to the given question or query**.

---

### 🧠 **Definition:**

> **Context Precision** = (Number of relevant context chunks retrieved) / (Total number of context chunks retrieved)

It's used to evaluate how well your **retriever** is performing—not the generator.

---

### ✅ **1. Context Precision WITH Reference**

This is **evaluated using a known ground truth reference answer**.

#### 📊 How it's calculated:

* You have a **reference answer** (ideal or ground truth answer).
* You compare **each retrieved context chunk** to see if it **supports or overlaps with** the reference answer.
* **Precision** = relevant chunks / total retrieved chunks

🔍 **Use Case**: When you’re doing **benchmarking or testing**, and you know the correct answer.

#### 🧪 Example:

* **Question**: "What is the capital of France?"
* **Reference Answer**: "Paris"
* **Retrieved Chunks**:

  1. "Paris is the capital of France" ✅ (relevant)
  2. "France is in Europe" ❌ (not relevant)
  3. "Eiffel Tower is in Paris" ✅ (relevant)

**Precision** = 2 relevant chunks / 3 total chunks = **0.67**

---

### 🚫 **2. Context Precision WITHOUT Reference**

This is **used when you don’t have a known answer**. Instead, it measures **semantic similarity between the query and the context**.

#### 🔍 How it's done:

* Use a **similarity function** (like cosine similarity between embeddings of the query and each chunk).
* Set a threshold (e.g., 0.8) to decide if a chunk is relevant.
* Then count how many retrieved chunks meet or exceed that similarity threshold.

#### 📊 Formula:

> Precision = # of retrieved chunks with similarity > threshold / total retrieved chunks

---

### 🧠 Summary Table:

| Metric Variant    | Requires Reference Answer? | Relevance Check         | Use Case                  |
| ----------------- | -------------------------- | ----------------------- | ------------------------- |
| With Reference    | ✅ Yes                      | Matches/supports answer | Ground truth evaluation   |
| Without Reference | ❌ No                       | Similarity to query     | Real-time/self-supervised |

---

Would you like Python code examples for either variant (e.g., using `sentence-transformers`)?
