# matchemeplease

An quick and dirty experimentation to match candidates with job descriptions based on their profiles

## Idea

We want to match candidates with job description with as limited input as possible and a novel way of matching.
Here is our idea:

- Given a Linkedin profile dump
- Generate an "ideal" job description
- Perform a semantic search against the scraped job descriptions

## Run it

1. Install the requirements
2. Load the data into the vector DB running `load_data.py`
3. Tweak the data you want to match with in the file and run `match.py`

## Details

-`descriptions.xlsx` contains a bunch of job descriptions we grabbed from KeyValues.com

- `profiles` contains Linkeding profile pdf dumps
- `results` contains the match results
