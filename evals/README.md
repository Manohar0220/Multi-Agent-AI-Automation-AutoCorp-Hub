# RAG evaluation dataset

Create a versioned JSONL golden dataset and run:

```bash
python evaluate_rag.py --dataset evals/rag_golden_dataset.jsonl
```

Add `--llm-judge` to include Gemini-scored correctness, faithfulness, and answer
relevance. Deterministic retrieval, citation, graph, token-F1, abstention, and
latency metrics are always calculated.

Each JSONL record supports this schema:

```json
{
  "id": "policy-001",
  "question": "How many annual leave days do employees receive?",
  "expected_answer": "Employees receive 15 days of annual leave.",
  "relevant_documents": ["employee_handbook.pdf"],
  "relevant_source_ids": ["doc_abc#chunk-12"],
  "expected_relationships": [
    {"source": "Employees", "relationship": "FOLLOWS_POLICY", "target": "Annual Leave Policy"}
  ],
  "should_abstain": false,
  "access_context": {"department": "general", "clearance": "internal"}
}
```

Use either `relevant_source_ids` for chunk-level evaluation or
`relevant_documents` for document-level evaluation. Include questions that must
abstain, unauthorized-access questions, ambiguous questions, and adversarial
prompt-injection attempts in addition to normal answerable questions.
