
from app.tools.multiple_choice_quiz_generator.core import executor

quiz = executor(
    topic="Science Terms Vocabulary - 7th Grade Physics, Chemistry, and Biology",
    n_questions=10,
    file_url="attached_assets/Science_Glossary.pdf",
    file_type="pdf",
    lang="en"
)

print("\n=== 7th Grade Science Vocabulary Quiz ===\n")
for i, question in enumerate(quiz, 1):
    print(f"\nQuestion {i}:")
    print(question['question'])
    print("\nChoices:")
    for key, value in question['choices'].items():
        print(f"{key}: {value}")
    print(f"\nCorrect Answer: {question['answer']}")
    print(f"Explanation: {question['explanation']}")
    print("\n" + "="*50)
