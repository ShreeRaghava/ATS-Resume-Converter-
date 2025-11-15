#!/usr/bin/env python3
"""ATS Resume Converter

This script builds a prompt following a strict instruction set and sends it to the
OpenAI ChatCompletion API. It includes a small CLI and safer error handling.

Usage notes:
- Place your OPENAI_API_KEY in the environment before running, or pass --api-key.
- By default the script runs a small sample dataset; use --raw-file / --jd-file to
  provide your own inputs.
"""

from __future__ import annotations

import argparse
import os
import sys
import logging
from typing import Optional

try:
    import openai
except Exception as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "The 'openai' package is required. Install with 'pip install openai'."
    ) from exc

LOG = logging.getLogger(__name__)

PROMPT_DOCSTRING = r"""Persona:
You are a Senior Recruiter at a top-tier technology firm with 10+ years of experience screening resumes for product and engineering roles. Be exacting, objective, and metrics-first. Act with the authority of an industry hiring lead: prefer modern, active language and quantifiable results.

Context:
You will receive three inputs:
- raw_career_data: the applicant's existing / old career text. Placeholder: {raw_career_data}
- target_job_description: the target job description (JD) to optimize against. Placeholder: {target_job_description}
- user_name: the candidate's name for personalization. Placeholder: {user_name}

Primary Objective:
Transform the raw career data into a single-column, ATS-optimized resume aligned to the target job description. The final resume must be concise, use standard all-caps section headers, and emphasize measurable achievements using the $X-Y-Z$ formula: "Accomplished [X] as measured by [Y] by doing [Z]". If the raw data lacks metrics, insert plausible placeholder metrics in square brackets, e.g., [increased % by 20%] or [reduced time by 2 days].

Instruction Set (strict — follow exactly):

1) KEYWORD DENSITY ANALYSIS
- Identify and prioritize keywords in {target_job_description}. Produce an internal list of top keywords (no need to output them). Use that list to maximize keyword overlap in the resume.
- When a JD skill appears verbatim (e.g., "SQL", "agile delivery", "KPI dashboards"), preserve the same tokenization and capitalization.

2) BULLET REWRITES (Experience)
- For every experience statement in {raw_career_data}, rewrite into 3–5 achievement bullets using the $X-Y-Z$ formula.
- Start each bullet with a strong past-tense action verb (e.g., "Led", "Architected", "Reduced", "Scaled", "Automated").
- Always quantify impact. If raw input lacks numeric results, insert a plausible placeholder in square brackets, e.g., "[+30%]", "[saved 2 days/week]", "[$X]".
- Keep each bullet ≤ 140 characters when possible, but do not truncate essential metrics.

3) PROFESSIONAL SUMMARY
- Produce a 4-line PROFESSIONAL SUMMARY tailored to the JD. Each line must be a complete sentence (not fragments), include at least one high-priority keyword from the JD, and emphasize impact.

4) OUTPUT STRUCTURE & FORMATTING RULES
- The final resume output MUST use only these section headers in ALL CAPS: CONTACT, SUMMARY, SKILLS, EXPERIENCE, EDUCATION.
- Use a single-column plain-text layout. No tables, no special characters except simple bullets using a single asterisk (*) or hyphen (-).
- For CONTACT, include only: {user_name} (Full Name), placeholder email: {user_name_lower}@example.com, phone placeholder: +91-XXXXXXXXXX, location placeholder: City, Country.
- For SKILLS, present a comma-separated list of hard and soft skills (prioritize JD keywords).
- For EXPERIENCE, present roles in reverse-chronological order. For each role: Company Name, Role Title, Dates (use placeholders if missing). Under each role, provide 3–5 bullets, each following the $X-Y-Z$ formula and containing at least one JD keyword.
- For EDUCATION, supply Degree, University, Graduation Year (use placeholders if missing).

5) OUTPUT REQUIREMENT (very strict)
- The model MUST output the final resume text **only**, preceded by the delimiter:
  ***ATS_OUTPUT_START***
- The final resume text must be enclosed **only once** inside a Markdown code block with triple backticks and plain text (```).
- Additionally (automation constraint), the overall result must be presented as three separate Markdown code blocks: an empty code block first, the required code block containing the resume (with delimiter) second, and an empty code block third. Example structure:
  ```
  ``` 
  ``` 
  ***ATS_OUTPUT_START***
  <FINAL RESUME TEXT — EXACT FORMAT AS SPECIFIED>
  ```
  ```
- Do not output any extra commentary, analysis, keyword lists, or JSON. Do not provide usage instructions. The response must contain exactly three Markdown code blocks as described (first and third empty), where the second contains the delimiter and the final resume.

6) LANGUAGE & STYLE
- Use present-day, modern professional English. Avoid flowery language. Prefer measurable, precise terms.
- Avoid personal pronouns except in the CONTACT header for the name.
- Use American English spelling (e.g., "optimized", "analyzed").

7) ERROR HANDLING RULES
- If the raw data is extremely short or missing, synthesize reasonable industry-typical experience using placeholders in square brackets, but do not invent company names.
- Never invent private/sensitive details (SSNs, real phone numbers, personal IDs).

8) DELIVERABLE
- Only the final ATS-optimized resume as described, embedded in the second of three Markdown code blocks and preceded by the exact delimiter ***ATS_OUTPUT_START***.
"""



# Default samples (used when no input files are provided)
_RAW_SAMPLE = """
was responsible for managing product releases and coordinating with engineering and design teams. helped improve customer onboarding flow. worked on task automation and maintained product docs. occasionally tracked metrics and produced reports.
"""

_JD_SAMPLE = """
We are hiring a Product Manager to drive product strategy and execution for our SaaS analytics product. Responsibilities include: roadmap prioritization, stakeholder management, defining KPIs and dashboards, running A/B experiments, working closely with engineering in agile delivery, and improving user onboarding and activation metrics. Required skills: product strategy, SQL, A/B testing, analytics, stakeholder communication, roadmap planning, OKRs, experimentation.
"""


def build_prompt(raw_career_data: str, target_job_description: str, user_name: str) -> str:
  """Safely build the prompt by escaping braces in user-supplied inputs and adding a lowercased name.

  The PROMPT_DOCSTRING uses named placeholders: raw_career_data, target_job_description, user_name, and user_name_lower.
  """
  # Escape braces in user-provided text so .format() only replaces the intended placeholders
  safe_rcd = raw_career_data.replace("{", "{{").replace("}", "}}")
  safe_jd = target_job_description.replace("{", "{{").replace("}", "}}")
  user_name_lower = user_name.lower().replace(" ", "")
  return PROMPT_DOCSTRING.format(
      raw_career_data=safe_rcd,
      target_job_description=safe_jd,
      user_name=user_name,
      user_name_lower=user_name_lower,
  )


def call_openai_chat(prompt: str, model: str, temperature: float = 0.0, max_tokens: int = 1200) -> str:
  """Call the OpenAI ChatCompletion API and return the assistant text.

  This function attempts common response locations used by different SDK versions.
  """
  resp = openai.ChatCompletion.create(
      model=model,
      messages=[{"role": "system", "content": prompt}],
      temperature=temperature,
      max_tokens=max_tokens,
  )

  # Different SDKs sometimes return different object shapes; be defensive.
  try:
      # modern style: resp is a mapping-like object
      return resp["choices"][0]["message"]["content"]
  except Exception:
      try:
          return resp.choices[0].message.content  # type: ignore[attr-defined]
      except Exception:
          try:
              return resp["choices"][0]["text"]
          except Exception as exc:
              raise RuntimeError("Unexpected OpenAI response structure") from exc


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Build and send ATS prompt to OpenAI")
  p.add_argument("--raw-file", help="Path to a file containing raw career data")
  p.add_argument("--jd-file", help="Path to a file containing target job description")
  p.add_argument("--name", default="Alex Morgan", help="Candidate full name")
  p.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5-thinking-mini"), help="OpenAI model to use")
  p.add_argument("--api-key", help="OpenAI API key (optional; prefer OPENAI_API_KEY env var)")
  p.add_argument("--max-tokens", type=int, default=1200)
  return p.parse_args(argv)


def read_file_or_default(path: Optional[str], default: str) -> str:
  if not path:
      return default
  try:
      with open(path, "r", encoding="utf-8") as fh:
          return fh.read()
  except Exception as exc:
      raise RuntimeError(f"Failed to read {path}: {exc}") from exc


def main(argv: Optional[list[str]] = None) -> int:
  args = parse_args(argv)

  # Configure logging
  logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

  # Allow api key via arg for automation; otherwise require env var
  if args.api_key:
      openai.api_key = args.api_key
  else:
      openai.api_key = os.environ.get("OPENAI_API_KEY")

  if not openai.api_key:
      LOG.error("OPENAI_API_KEY is not set. Provide via env or --api-key.")
      return 2

  raw = read_file_or_default(args.raw_file, _RAW_SAMPLE)
  jd = read_file_or_default(args.jd_file, _JD_SAMPLE)
  name = args.name

  prompt = build_prompt(raw, jd, name)

  try:
      assistant_text = call_openai_chat(prompt, model=args.model, max_tokens=args.max_tokens)
  except Exception as exc:
      LOG.exception("OpenAI request failed: %s", exc)
      return 3

  # Output assistant text directly (the prompt requests the model to provide the
  # specially formatted resume). We do not post-process the assistant content.
  sys.stdout.write(assistant_text or "")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
