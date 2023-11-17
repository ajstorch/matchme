import json
from dataclasses import dataclass
from enum import Enum
from typing import List
from pathlib import Path
import os
from pypdf import PdfReader
from langchain.llms import OpenAI
from vector_db import singleton_client

from langchain.prompts import PromptTemplate


BASE_PATH = Path(__file__).parent
OPEN_AI_KEY = "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"
os.environ["OPENAI_API_KEY"] = OPEN_AI_KEY

job_similarity_template = PromptTemplate.from_template(
    """
    Given that this represents the ideal job description for a candidate

    "Ideal Job Description":
    ```
    {ideal_job_description}
    ```

    And that this is a job description for a company

    "Job Description":
    ```
    {company_job_description}
    ```

    List 3 to 6 exercpts from the company job description that we think are relevant for the candidate.
    Use the following format:

        ```
        [company_job_description_excerpt, ...]
        ```
    """
)


def explain_similarities(ideal_job_description, company_job_description):
    prompt = job_similarity_template.format(
        ideal_job_description=ideal_job_description,
        company_job_description=company_job_description,
    )
    llm = OpenAI(
        openai_api_key=OPEN_AI_KEY,
        model_name="gpt-4",
        temperature=0.3,
    )
    result = llm(prompt)
    return json.loads(result)


def pretty_print_array(array):
    print(array)
    return "\n".join([f"- {item}" for item in array])


def remove_leading_whitespace(text):
    return "\n".join([line.strip() for line in text.split("\n")])


@dataclass
class MatchResults:
    ideal_job_descriptions: str
    ideal_company_culture: str
    job_descriptions: List[str]

    def dumps(self):
        excerpts = explain_similarities(
            ideal_job_description=self.ideal_job_descriptions,
            company_job_description=self.job_descriptions[0],
        )
        results = f"""
        "Here is a job we found for you":
        XXXXXXXXXXXXXXXXXXXX
        ```
        {self.job_descriptions[0].page_content}
        ```
        XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        "We think that the following excerpts from the job description are relevant for you!":
        ```
        {pretty_print_array(excerpts)}
        ```
        """

        return remove_leading_whitespace(results)

    def dumps_to_file(self, filename):
        path = BASE_PATH / "results" / filename
        # If the file does not exist, create it
        if not path.exists():
            path.touch()
        with open(path, "w") as f:
            f.write(self.dumps())


job_description_template = PromptTemplate.from_template(
    """
    You are a profesionnal carreer coach that is helping a client to find a job.
    Given the following information about the client:

    "Resume":
    ```
    {resume}
    ```
    Write the ideal Job Description for this client formatted in Markdown:
    """
)

company_culture_template = PromptTemplate.from_template(
    """
    You are a profesionnal carreer coach that is helping a client to find a job.
    Given the following information about the client:

    "Resume":
    ```
    {resume}
    ```
    Given that a blurb about a company culture would look like:
    ```
    Our mission is to democratize software creation by enabling anyone to build the tools that meet their needs.
    We believe software is the most important innovation of the past century, but that its potency as a medium for economic value creation and creative expression remains inaccessible to most. (People typically experience a finished software product rather than software as a medium.) We believe customer success isn’t the responsibility of a single team: instead, we all own this. In everything we do, we start with the customer and work backwards. When our customers succeed, we succeed. As engineers, we know how empowering it can be to create software, and we intend to bring this power to everyone.
    Instead of designing features for narrow use cases, we create fundamental building blocks that can be assembled to model any workflow. With our next-generation app platform, teams easily develop and deploy flexible and engaging apps that power critical workflows for teams. Our users can build powerful apps using our App designer, without requiring any special technical skills. Our platform makes it effortless to create an easy to use UI, powerful workflows, and scalable data structures. We strive to make complexity as accessible as possible, sweating the details of our user interface and creating the best possible experience even when it means more work up front.
    ```
    Write a blurb that describes the ideal company culture for this client formatted in Markdown
    """
)


class ItemTypes(Enum):
    JOB_DESCRIPTION = "job_description"
    COMPANY_CULTURE = "company_culture"


TEMPLATE_FOR_ITEM_TYPE = {
    ItemTypes.JOB_DESCRIPTION: job_description_template,
    ItemTypes.COMPANY_CULTURE: company_culture_template,
}


def generate_ideal_item(profile, item_type, **kwargs):
    resume = profile["resume"]
    template = TEMPLATE_FOR_ITEM_TYPE[item_type]
    prompt = template.format(resume=resume)
    llm = OpenAI(
        openai_api_key=OPEN_AI_KEY,
        model_name="gpt-4",
        temperature=0.3,
    )
    result = llm(prompt)
    return result


def generate_ideal_job_description(resume, **kwargs):
    prompt = job_description_template.format(resume=resume)
    llm = OpenAI(
        openai_api_key=OPEN_AI_KEY,
        model_name="gpt-4",
        temperature=0.3,
    )
    result = llm(prompt)
    return result


def extract_resume(user_profile):
    """
    Linkedin?
    PDF extraction
    """
    file_path = f"{user_profile['slug']}.pdf"
    resume_path = BASE_PATH / "profiles" / file_path
    pdf_reader = PdfReader(resume_path)
    full_extracted_text = ""
    for page in pdf_reader.pages:
        full_extracted_text += page.extract_text()

    return full_extracted_text


def extract_pymetrics_results(user_profile):
    """
    Pymetrics API
    """
    return "These are the pymetrics results"


def find_top_matches(user_profile) -> MatchResults:
    """
    Use embeddings to find top matches
    """
    ideal_job_description = generate_ideal_job_description(**user_profile)
    ideal_company_culture = generate_ideal_item(
        profile=user_profile,
        item_type=ItemTypes.COMPANY_CULTURE,
    )
    vector_db_client = singleton_client()
    documents = vector_db_client.similarity_search(query=ideal_job_description, k=1)
    return MatchResults(
        ideal_job_descriptions=ideal_job_description,
        ideal_company_culture=ideal_company_culture,
        job_descriptions=documents,
    )


USER_PROFILES = [
    {
        "name": "Anna Julia Storch",
        "slug": "anna_julia",
        "skills": ["Excel", "SQL"],
        "interests": [
            "Sports",
        ],
        "values": ["Teamwork", "Collaboration"],
    },
    {
        "name": "Kelly Byrd",
        "slug": "kelly_byrd",
        "skills": ["Python", "Javascript", "React", "NodeJS", "MongoDB"],
        "interests": [
            "Machine Learning",
            "Artificial Intelligence",
            "Data Science",
            "Computer Vision",
        ],
        "values": ["Honesty"],
    },
]


if __name__ == "__main__":
    user_profile = USER_PROFILES[1]
    user_profile["resume"] = extract_resume(user_profile)
    results = find_top_matches(user_profile)
    results.dumps_to_file(f"{user_profile['slug']}.pdf")
    print("Match done")
