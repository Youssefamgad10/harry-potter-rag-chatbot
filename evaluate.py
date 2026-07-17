from rag_core import build_pipeline, ask_question

print("Building pipeline (this takes a moment)...")
pipeline = build_pipeline()
print("Pipeline ready.\n")

test_set = [
    ("What does Harry's uncle do for work?", "director of a firm called Grunnings, which makes drills"),
    ("What street do the Dursleys live on?", "Privet Drive"),
    ("What is the name of Harry's school?", "Hogwarts"),
    ("What house is Harry sorted into?", "Gryffindor"),
    ("Who is the headmaster of Hogwarts?", "Albus Dumbledore"),
    ("What position does Harry play in Quidditch?", "Seeker"),
    ("What is the three-headed dog's name?", "Fluffy"),
    ("Who teaches Potions?", "Professor Snape"),
    ("Who tries to steal the Sorcerer's Stone?", "Professor Quirrell, possessed by Voldemort"),
    ("Which house wins the House Cup?", "Gryffindor"),
]

for question, expected in test_set:
    answer, _ = ask_question(pipeline, question)
    print(f"Q: {question}")
    print(f"Expected: {expected}")
    print(f"Got: {answer}\n")