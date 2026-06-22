# Running Legal Research Agent Benchmark

Our Legal Research Agent benchmark evaluates LLMs on their ability to use tools to research and answer complex legal questions about statutes, regulations, and case law across federal and state jurisdictions.

The agent has access to the following tools:

- `web_search`: Search the web for information (via Tavily)
- `parse_html_page`: Fetch a web page (e.g. a statute, regulation, or opinion), strip it to text, and store it in the agent's data storage
- `retrieve_information`: Query stored documents with the active LLM using `{{key}}` references
- `courtlistener_search`: Search and retrieve U.S. federal and state court opinions, dockets, judges, and citations via [CourtListener](https://www.courtlistener.com/)

For more details on the benchmark, please refer to our [public website](https://www.vals.ai/benchmarks/legal-research).

## Set up

### Dependencies

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management. Then run:

```
make install
source .venv/bin/activate
```

### Platform

Access to the Vals platform is gated and requires approval. Please reach out to us at [vals.ai](https://www.vals.ai/) to request access.

Once approved, make an account on [platform.vals.ai](https://www.platform.vals.ai/auth) with your company email address. Go to the admin page and create a new API key for yourself.

### Environment Variables

Create a `.env` file in the root of the project and add the following:

```
VALS_API_KEY=<api_key>

# LLM API Keys (only set the ones you plan on using)
OPENAI_API_KEY=<openai_api_key>
ANTHROPIC_API_KEY=<anthropic_api_key>
GOOGLE_API_KEY=<google_api_key>
ETC_API_KEY=<etc_api_key>

# Tool API Keys
TAVILY_API_KEY=<tavily_api_key>
COURTLISTENER_API_KEY=<courtlistener_api_key>
```

You can create a Tavily API key [here](https://tavily.com/), and a CourtListener API key [here](https://www.courtlistener.com/help/api/rest/).

The `.env` takes precedence over set environment variables.

## Running the benchmark

For a list of command line options, run `legal-research --help`

To run, for example, a single question on google/gemini-3.1-pro-preview:

```
legal-research --questions "What are the elements of a 42 U.S.C. § 1983 claim?" --model google/gemini-3.1-pro-preview
```

You can specify multiple questions at once:

```
legal-research --questions "What are the elements of a § 1983 claim?" "What is the standard for qualified immunity?"
```

You can also specify a list of questions in a file. The benchmark's sample questions live in `data/public.json`:

```
legal-research --question-file data/public.json
```

`--question-file` accepts a `.json` file (a list of question strings, a list of objects with a `question` field, or an object with a top-level `questions` list) or a `.txt` file with one question per line.

The default configuration is the one we used to run the benchmark.

### Tools

By default the agent gets the full tool set. Restrict it with `--tools`:

```
legal-research --question-file data/public.json --tools web_search parse_html_page retrieve_information courtlistener_search
```

### List of Models

A list of available models can be found at our [model library](https://github.com/vals-ai/model-library/blob/main/model_library/config/all_models.json), and also by running `make browse-models` in the model library repository.

To run your own harness or model, just modify the agent construction in `legal_agent/get_agent.py` as needed. To see the full documentation on how the SDK works, visit [our docs](https://docs.vals.ai/sdk/running_suites).

## Logs

The agent writes detailed logs to the `logs/` directory. Each run creates per-question log files containing tool usage, token counts, and error tracking, and a `results.json` summary alongside them.
