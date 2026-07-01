# LinkedIn Easy Apply Agent

An autonomous agent that finds LinkedIn **Easy Apply** jobs matching your search and applies
to them for you, filling the application form (including screening questions) from your own
instructions, and submitting. It saves a markdown record of every confirmed application.

Built as a tool-calling agent with [LangChain `create_agent`](https://docs.langchain.com/oss/python/langchain/agents),
DeepSeek, and async Playwright, the LLM drives a real browser through tools.

## Why

Applying to jobs one by one on LinkedIn is slow and repetitive, and the same screening
questions come up again and again. This agent automates the mechanical part: it searches,
opens each new job, answers the questions from a policy you control, and submits, so you
can cover far more of the market in far less time. It only applies to jobs that match your
filter, and never touches ones you've already applied to.

## Features

- **Filter-driven**: keyword, location, recency, remote, and Easy-Apply come from `instructions.md`.
- **Answers screening questions**: from an answering policy + screening answers you set.
- **Skips already-applied jobs**: reads the "Applied" state from the results list.
- **Handles LinkedIn's interstitials**: proceeds past "safety reminder" / "Continue applying" pop-ups.
- **Saves a record**: one markdown file per confirmed application, under `data/<keyword>-<location>/`.

## Requirements

- **Python 3.10+** (tested on 3.13)
- **A DeepSeek API key**: set as `DEEPSEEK_API_KEY` in a `.env` file
- **A LinkedIn account**: logged in once in the browser profile the agent opens (it runs as you)
- **Your resume**: placed as `resume.pdf` in this folder
- **Python packages:** `langchain`, `langchain-deepseek`, `langgraph`, `langchain-core`, `playwright`, `python-dotenv`
- **Playwright's Chromium browser**: installed with `playwright install chromium`
- An internet connection

## Setup

```bash
pip install -r requirements.txt
playwright install chromium
cp .env.example .env                 # then add your DEEPSEEK_API_KEY
cp instructions.example.md instructions.md   # then fill in your filter + answers
# put your resume in this folder as resume.pdf
```

Run the agent once and log into LinkedIn in the Chromium window it opens, the session
persists for future runs.

## Usage

```bash
python agent-linkedin-easy-apply.py
```

## Configuration: `instructions.md`

```
## Search filter
keywords: react
location: United States
date_posted: 1 week
easy_apply: true
remote: true
```

The `## Answering policy` and `## Screening question answers` sections tell the agent how to
answer the application questions. See `instructions.example.md`.

## Output

```
data/<keyword>-<location>/<job_id>-<company>.md
```

## How it works

A single agent loop: `list_jobs → open_job → read_job → open_easy_apply → read_form → fill →
submit_application`. The tools handle the browser; the LLM decides what to open, how to
answer each field, and when the application is submitted.

## Notes

- **It submits real applications.** Review your `instructions.md` before running.
- Nothing personal is committed: `.env`, `resume.pdf`, `instructions.md`, `data/`, and the
  browser profile are all git-ignored.
