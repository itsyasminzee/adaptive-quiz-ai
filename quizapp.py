import streamlit as st
import json
import os
from dotenv import load_dotenv
from openai import OpenAI
import PyPDF2

load_dotenv()

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_API_KEY"),
)

def extract_pdf_text(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""

    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"

    return text.strip()


def fetch_questions(text_content, quiz_level, num_questions=10):
    num_questions = min(num_questions, 50)

    RESPONSE_JSON = {
        "acqs": [
            {
                "acq": f"Sample question {i+1}",
                "options": {
                    "a": "choice1",
                    "b": "choice2",
                    "c": "choice3",
                    "d": "choice4"
                },
                "correct": "a",
            } for i in range(num_questions)
        ]
    }

    PROMPT = f"""
Text:
{text_content}

Generate {num_questions} multiple choice questions.
Difficulty: {quiz_level}

Return ONLY valid JSON in this format:
{json.dumps(RESPONSE_JSON, indent=2)}
"""

    try:
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=[{"role": "user", "content": PROMPT}],
            temperature=0.3,
            max_tokens=4000,
        )

        result = response.choices[0].message.content.strip()
        parsed = json.loads(result)
        return parsed.get("acqs", RESPONSE_JSON["acqs"])

    except Exception as e:
        st.error(f"Quiz generation failed: {e}")
        return RESPONSE_JSON["acqs"]


def generate_ai_feedback(results_summary):
    prompt = f"""
Write the feedback in SECOND PERSON (use "you").

You must respond in this structure:

1. Performance Analysis (talk directly to the student using "you")
2. Strengths (what you did well)
3. Weak Areas (general patterns of mistakes, NOT specific questions)
4. Improvement Advice (what you should focus on next)
5. Motivation (short encouraging closing message)

Rules:
- Always use "you"
- Do NOT refer to the student as "he", "she", or "the student"
- Do NOT mention specific questions or answers
- Be clear, supportive, and structured
- Keep it concise but insightful
- Motivation should be short and positive

Student Performance Data:
{json.dumps(results_summary, indent=2)}
"""

    try:
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=1200,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Feedback error: {e}"

def main():
    st.title("📄 AI Quiz Generator App")

    uploaded_pdf = st.file_uploader("Upload PDF (optional)", type=["pdf"])

    pdf_text = ""
    if uploaded_pdf:
        pdf_text = extract_pdf_text(uploaded_pdf)
        st.success("PDF loaded successfully!")

    text_content = st.text_area(
        "Or paste text here (PDF overrides this):",
        value=pdf_text
    )

    quiz_level = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"]).lower()

    num_questions = st.number_input(
        "Number of questions",
        min_value=1,
        max_value=50,
        value=10
    )

    if "questions" not in st.session_state:
        st.session_state.questions = []
        st.session_state.answers = []

    # ---------------- GENERATE QUIZ ----------------
    if st.button("Generate Quiz"):
        final_text = pdf_text if uploaded_pdf else text_content

        if not final_text.strip():
            st.warning("Please provide input text or PDF.")
        else:
            st.session_state.questions = fetch_questions(
                final_text, quiz_level, num_questions
            )
            st.session_state.answers = [None] * len(st.session_state.questions)

    # ---------------- QUIZ ----------------
    if st.session_state.questions:
        st.header("📝 Answer Questions")

        for i, q in enumerate(st.session_state.questions):
            options = list(q["options"].values())

            st.session_state.answers[i] = st.radio(
                f"{i+1}. {q['acq']}",
                options,
                index=None,
                key=f"q{i}"
            )

    # ---------------- SUBMIT ----------------
    if st.session_state.questions:

        if st.button("Submit"):

            total = len(st.session_state.questions)
            marks = 0

            results_summary = []
            wrong_questions = []

            for i, q in enumerate(st.session_state.questions):

                selected = st.session_state.answers[i]
                correct_letter = q["correct"]
                correct = q["options"][correct_letter]

                is_correct = selected == correct

                if is_correct:
                    marks += 1
                else:
                    wrong_questions.append(q["acq"])

                results_summary.append({
                    "question": q["acq"],
                    "your_answer": selected,
                    "correct_answer": correct,
                    "is_correct": is_correct
                })

            accuracy = (marks / total) * 100 if total else 0

            # ---------------- DASHBOARD ----------------
            st.header("📊 Results Dashboard")

            col1, col2, col3 = st.columns(3)
            col1.metric("Score", f"{marks}/{total}")
            col2.metric("Accuracy", f"{accuracy:.1f}%")
            col3.metric("Wrong", len(wrong_questions))

            st.progress(int(accuracy))

            st.divider()

            # ---------------- REVIEW ----------------
            st.subheader("📝 Review Answers")

            for i, r in enumerate(results_summary):
                icon = "✅" if r["is_correct"] else "❌"

                with st.expander(f"{icon} Question {i+1}"):
                    st.write("**Question:**", r["question"])
                    st.write("**Your Answer:**", r["your_answer"])
                    st.write("**Correct Answer:**", r["correct_answer"])

                    if r["is_correct"]:
                        st.success("Correct")
                    else:
                        st.error("Incorrect")

            st.divider()

            # ---------------- RETRY WRONG ----------------
            if wrong_questions:
                if st.button("🔁 Retry Wrong Answers"):
                    filtered = [
                        q for q in st.session_state.questions
                        if q["acq"] in wrong_questions
                    ]

                    st.session_state.questions = filtered
                    st.session_state.answers = [None] * len(filtered)

                    st.rerun()

            st.divider()

            # ---------------- FEEDBACK ----------------
            st.header("🧠 Feedback")

            with st.spinner("Analyzing your performance..."):
                feedback = generate_ai_feedback(results_summary)

            st.info(feedback)


if __name__ == "__main__":
    main()