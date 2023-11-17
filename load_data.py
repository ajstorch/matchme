from pathlib import Path
import pandas as pd
from dataclasses import dataclass
from vector_db import singleton_client
import os


BASE_PATH = Path(__file__).parent
OPEN_AI_KEY = "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"
os.environ["OPENAI_API_KEY"] = OPEN_AI_KEY


@dataclass
class JobDescription:
    company: str
    description: str
    culture: str


def load_data():
    """Load matching data from excel"""
    data_path = BASE_PATH / "data" / "descriptions.xlsx"
    df = pd.read_excel(data_path)
    vector_client = singleton_client()
    for i, row in df.iterrows():
        job_description = JobDescription(
            company=row["Company"],
            description=row["Job Description"],
            culture=row["Culture Description"],
        )
        if isinstance(job_description.description, str):
            vector_client.add_texts(
                texts=[job_description.description],
                metadatas=[
                    {
                        "company": job_description.company,
                    }
                ],
            )


if __name__ == "__main__":
    load_data()
    print("Done loading data")
