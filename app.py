import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.wikipedia import WikipediaTools
import google.generativeai as genai
import os
import tempfile
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# Load API key
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

# Streamlit page setup
st.set_page_config(
    page_title="Resume Reviewer Agent",
    page_icon="🧾",
    layout="wide"
)

st.title("🧾 AI Resume Reviewer Agent")
st.header("Powered by Gemini 2.0 Flash Exp")

# Initialize a single, multi-skilled agent
resume_review_agent = Agent(
    name="ResumeReviewAgent",
    role=(
        "A senior career advisor and resume expert. "
        "You analyze resumes, compare them to job descriptions, identify skill gaps, "
        "suggest improvements, and generate cover letters when asked."
    ),
    model=Gemini(id="gemini-2.0-flash-exp"),
    tools=[DuckDuckGo(), WikipediaTools()],
    markdown=True
)

# File upload
pdf_file = st.file_uploader("📄 Upload your Resume (PDF or TXT)", type=["pdf", "txt"])
jd_text = st.text_area("💼 Paste Job Description here", height=300)

# Extract text
def extract_pdf_text(file_path):
    reader = PdfReader(file_path)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

def extract_txt_text(file):
    return file.read().decode("utf-8")

resume_text = ""
if pdf_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_file.read())
        pdf_path = temp_pdf.name

    if pdf_path.endswith(".pdf"):
        resume_text = extract_pdf_text(pdf_path)
    else:
        resume_text = extract_txt_text(pdf_file)

# User query
user_query = st.text_area(
    "What would you like the AI to review in your resume?",
    placeholder="Examples:\n- How well does my resume match the JD?\n- Suggest improvements.\n- Generate a tailored cover letter.",
    help="Ask the AI to match your resume, find gaps, or even create a cover letter."
)

# Analyze button
if st.button("🔍 Analyze Resume"):
    if not resume_text:
        st.warning("Please upload your resume first.")
    elif not user_query:
        st.warning("Please enter a query for the AI.")
    else:
        with st.spinner("Analyzing your resume..."):

            prompt = f"""
            A user has submitted the following resume:

            {resume_text}

            Job Description (if provided):

            {jd_text}

            The user has asked:
            "{user_query}"

            Please:
            - Review the resume's strengths and areas for improvement.
            - Match it with the job description (if provided).
            - Identify any skill gaps or missing keywords.
            - Provide actionable advice.
            - Generate a personalized cover letter if requested.

            Format the response in markdown with headings and bullet points.
            """

            try:
                response = resume_review_agent.run(prompt)
                st.subheader("📋 AI Review Summary")
                st.markdown(response.content)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Styling text area height
st.markdown("""
    <style>
    .stTextArea textarea {
        height: 100px;
    }
    </style>
""", unsafe_allow_html=True)
