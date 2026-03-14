from autoreg.answering import AnswerGenerator, RAGSampleInput
g = AnswerGenerator(model_name="allenai/led-base-16384", max_new_tokens=64, device=-1)
sample = RAGSampleInput(query="What is photosynthesis?", retrieved_docs=[
    "Photosynthesis is the process by which plants convert light energy into chemical energy stored in glucose. It occurs in chloroplasts.",
    "Plants use chlorophyll to capture light. Light reactions generate ATP and NADPH.",
    "Calvin cycle fixes carbon dioxide into sugar."
])
print(g.generate_answer(sample))

