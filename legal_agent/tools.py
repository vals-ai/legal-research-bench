import json
import logging
import os
import re
from typing import Any

import aiohttp
from bs4 import BeautifulSoup
from model_library.agent import Tool, ToolOutput
from model_library.base import LLM
from tavily import AsyncTavilyClient

from .exceptions import (
    COURTLISTENER_MAX_TRIES,
    COURTLISTENER_MAX_VALUE,
    RetryExhaustedError,
    retry_http_errors,
)

MAX_END_DATE = "2026-05-10"
VALID_TOOLS = [
    "web_search",
    "retrieve_information",
    "parse_html_page",
    "courtlistener_search",
]
COURTLISTENER_RETRY_HTTP_STATUSES = (429, 503)


def _courtlistener_retry(*status_codes: int):
    """CourtListener-specific backoff (more tries, longer cap than default tools)."""
    return retry_http_errors(
        *status_codes,
        max_tries=COURTLISTENER_MAX_TRIES,
        max_time=None,
        max_value=COURTLISTENER_MAX_VALUE,
    )


class SubmitFinalResult(Tool):
    name = "submit_final_result"
    description = (
        "Submits the final answer to the user. You should include your final answer, as well as any necessary "
        "reasoning, justification, calculations, and explanation. Finally, you should provide any sources used to answer the question. "
        "You MUST use this tool to submit your final result. The user will not see your response if you do not use this tool to submit. "
        "You will not be able to continue working after this tool is called; the conversation will be ended."
    )
    parameters: dict[str, Any] = {
        "final_result": {
            "type": "string",
            "description": "The final result to submit to the agent",
        }
    }
    required: list[str] = ["final_result"]

    async def execute(
        self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger
    ) -> ToolOutput:
        try:
            final_result = args["final_result"]
            if not final_result:
                raise ValueError("Final result must not be empty")
            return ToolOutput(output=final_result, done=True)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Submission failed: {error_msg}")
            return ToolOutput(output=error_msg, error=error_msg, done=False)


class TavilyWebSearch(Tool):
    name = "web_search"
    description = "Search the public internet for information. Each result will contain a url, a title, and one excerpt taken directly from the page."
    parameters: dict[str, Any] = {
        "search_query": {
            "type": "string",
            "description": "The query to search for",
        },
        "start_date": {
            "type": "string",
            "description": "(optional) The start date for the search range for in the format YYYY-MM-DD",
        },
        "end_date": {
            "type": "string",
            "description": f"(optional) The end date to search range in the format YYYY-MM-DD. If the value is later than {MAX_END_DATE}, it will be set to {MAX_END_DATE}.",
        },
        "number_of_results": {
            "type": "integer",
            "description": "(optional) The number of search results to return.",
            "maximum": 20,
            "minimum": 1,
            "default": 10,
        },
    }
    required = ["search_query"]

    def __init__(self, tavily_api_key: str | None = None):
        if not tavily_api_key:
            tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            raise ValueError("TAVILY_API_KEY is not set")
        self.client = AsyncTavilyClient(api_key=tavily_api_key)

    @retry_http_errors(429)
    async def _execute_search(
        self,
        search_query: str,
        start_date: str | None = None,
        end_date: str = MAX_END_DATE,
        number_of_results: int = 10,
    ) -> list[dict[str, Any]]:
        DATE_REGEX = r"^\d{4}-\d{2}-\d{2}$"

        kwargs = {}

        if end_date:
            if not re.match(DATE_REGEX, end_date):
                raise ValueError(
                    f"Invalid end_date format: '{end_date}'. Expected YYYY-MM-DD."
                )

            if end_date > MAX_END_DATE:
                end_date = MAX_END_DATE

        if start_date:
            if not re.match(DATE_REGEX, start_date):
                raise ValueError(
                    f"Invalid start_date format: '{start_date}'. Expected YYYY-MM-DD."
                )
            if start_date > MAX_END_DATE:
                start_date = MAX_END_DATE
            if start_date > end_date:
                raise ValueError(
                    f"Parameter start_date '{start_date}' was set to a date that is later than end_date '{end_date}'"
                )

            kwargs["start_date"] = start_date

        response = await self.client.search(
            search_depth="fast",
            end_date=end_date,
            max_results=number_of_results,
            chunks_per_source=1,
            query=search_query,
            **kwargs,
        )

        return response.get("results", [])

    async def execute(
        self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger
    ) -> ToolOutput:
        try:
            results = await self._execute_search(**args)
            return ToolOutput(output=json.dumps(results, default=str))
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Web search failed: {error_msg}")
            return ToolOutput(output=error_msg, error=error_msg)


class ParseHtmlPage(Tool):
    name = "parse_html_page"
    description = (
        "This tool is used to parse the contents of an HTML page and save it to the agent's data storage system. "
        "The tool will retrieve the HTML page from the URL provided, then parse it from HTML to plain text. "
        "Finally, it will save it to the agent's data storage system under the key provided. "
        "You can use the retrieve_information tool to later retrieve information about the stored page."
    )
    parameters: dict[str, Any] = {
        "url": {"type": "string", "description": "The URL of the HTML page to parse"},
        "key": {
            "type": "string",
            "description": "The key to use when saving the result in the conversation's data storage.",
        },
    }
    required = ["url", "key"]

    @retry_http_errors(429, 503)
    async def _parse_html_page(self, url: str) -> str:
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=60),
                    headers={"User-Agent": "ValsAI/katrina@vals.ai"},
                ) as response:
                    response.raise_for_status()
                    raw_bytes = await response.read()
                    declared_charset = response.charset
            except Exception as e:
                if len(str(e)) == 0:
                    raise TimeoutError(
                        "Timeout error when parsing HTML page after 60 seconds. The URL might be blocked or the server is taking too long to respond."
                    )
                raise

        # Pass raw bytes so BeautifulSoup's UnicodeDammit can fall back from the
        # server-declared charset through <meta charset> and chardet to a lossy
        # decode, avoiding UnicodeDecodeError on non-UTF-8 pages.
        soup = BeautifulSoup(raw_bytes, "html.parser", from_encoding=declared_charset)
        for script_or_style in soup(["script", "style"]):
            _ = script_or_style.extract()

        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)

        return text

    async def _save_tool_output(
        self, output: str, key: str, state: dict[str, Any]
    ) -> str:
        if not output:
            raise ValueError("HTML output was empty")

        tool_result = ""
        if key in state:
            tool_result = "WARNING: The key already exists in the data storage. The new result overwrites the old one.\n"
        tool_result += (
            f"SUCCESS: The result has been saved to the data storage under the key: {key}."
            + "\n"
        )

        state[key] = output

        keys_list = "\n".join(state.keys())
        tool_result += (
            f"""
        The data_storage currently contains the following keys:
        {keys_list}
        """.strip()
            + "\n"
        )

        return tool_result

    async def execute(
        self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger
    ) -> ToolOutput:
        try:
            url = args["url"]
            key = args["key"]
            text_output = await self._parse_html_page(url)
            tool_result = await self._save_tool_output(text_output, key, state)
            return ToolOutput(output=tool_result)
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Parse HTML page failed: {error_msg}")
            return ToolOutput(output=error_msg, error=error_msg)


class RetrieveInformation(Tool):
    name = "retrieve_information"
    description = (
        "This tool allows you to retrieve data from previously saved documents from the agent's data storage system, by applying an LLM prompt to the stored document.\n"
        "\n"
        "To use the tool, you will need to provide a prompt. This prompt will include both the query to be sent to the LLM, "
        "as well as the keys of files you have previously saved to the data storage system.\n"
        "\n"
        'For example, if you want to analyze data stored under the key "financial_report", your prompt should look like the following:\n'
        '"Analyze the following financial report and extract the revenue figures: {{financial_report}}"\n'
        "\n"
        "The {{key_name}} will be replaced with the full text of the document stored under that key before the query is sent.\n"
        "\n"
        "IMPORTANT: Your prompt MUST include at least one key from the data storage using this exact format: {{key_name}}. "
        "If you don't use this exact format with double braces, the tool will fail to retrieve the information.\n"
        "\n"
        "You can also optionally only pass *a portion* of each document to the LLM, rather than the entire document. This can be used to avoid token limit errors or improve efficiency. "
        "To do so, use the input_character_ranges parameter to specify which portions of documents to extract. "
        'For example, if "financial_report" contains "Annual Report 2023" and you specify:  [{"key": "financial_report", "start": 1, "end": 6}], '
        'then only "nnual" will be inserted into the prompt (characters 1 through 5, as end is exclusive).'
    )
    parameters: dict[str, Any] = {
        "prompt": {
            "type": "string",
            "description": "The prompt that will be passed to the LLM. You MUST include at least one data storage key in the format {{key_name}} - for example: 'Summarize this 10-K filing: {{company_10k}}'. The content stored under each key will replace the {{key_name}} placeholder.",
        },
        "input_character_ranges": {
            "type": "array",
            "description": "An optional list of character range specifications for extracting only portions of documents. Each object should have 'key' (the document key), 'start' (start character index, inclusive), and 'end' (end character index, exclusive). By default, the full document is used if this parameter is not provided or if a key is not included in the list.",
            "items": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The document key from data storage",
                    },
                    "start": {
                        "type": "integer",
                        "description": "The starting character index (inclusive)",
                    },
                    "end": {
                        "type": "integer",
                        "description": "The ending character index (exclusive)",
                    },
                },
                "required": ["key", "start", "end"],
            },
        },
    }
    required = ["prompt"]

    def __init__(self, llm: LLM):
        self._llm = llm

    def _validate_inputs(
        self, prompt: str, input_character_ranges: list, state: dict[str, Any]
    ) -> dict[str, tuple[int, int]]:
        """Validate prompt placeholders, character ranges, and data storage keys. Returns the parsed ranges dict."""
        if not re.search(r"{{[^{}]+}}", prompt):
            raise ValueError(
                "ERROR: Your prompt must include at least one key from data storage in the format {{key_name}}. Please try again with the correct format. You can add documents to the data storage with parse_html_page."
            )

        ranges_dict = {}
        for range_spec in input_character_ranges:
            if not isinstance(range_spec, dict):
                raise ValueError(
                    "ERROR: Each item in input_character_ranges must be an object with 'key', 'start', and 'end' fields."
                )
            if (
                "key" not in range_spec
                or "start" not in range_spec
                or "end" not in range_spec
            ):
                raise ValueError(
                    "ERROR: Each range specification must have 'key', 'start', and 'end' fields."
                )
            ranges_dict[range_spec["key"]] = (range_spec["start"], range_spec["end"])

        keys = re.findall(r"{{([^{}]+)}}", prompt)
        keys_set = set(keys)

        for range_key in ranges_dict.keys():
            if range_key not in keys_set:
                raise ValueError(
                    f"ERROR: The key '{range_key}' is specified in input_character_ranges but is not referenced in the prompt. "
                    f"Keys in prompt: {', '.join(keys_set) if keys_set else '(none)'}"
                )

        for key in keys:
            if key not in state:
                raise KeyError(
                    f"ERROR: The key '{key}' was not found in the data storage. Available keys are: {', '.join(state.keys())}. Use the retrieve_information tool to add keys to the data storage."
                )

        return ranges_dict

    def _format_prompt(
        self,
        prompt: str,
        ranges_dict: dict[str, tuple[int, int]],
        state: dict[str, Any],
    ) -> str:
        """Substitute data storage content into prompt placeholders, applying character ranges."""
        keys = re.findall(r"{{([^{}]+)}}", prompt)
        formatted_data = {}

        for key in keys:
            doc_content = state[key]
            if key in ranges_dict:
                start_idx, end_idx = ranges_dict[key]
                formatted_data[key] = doc_content[start_idx:end_idx]
            else:
                formatted_data[key] = doc_content

        formatted_prompt = re.sub(r"{{([^{}]+)}}", r"{\1}", prompt)

        try:
            return formatted_prompt.format(**formatted_data)
        except KeyError as e:
            raise KeyError(
                f"ERROR: The key {str(e)} was not found in the data storage. Available keys are: {', '.join(state.keys())}"
            )

    async def execute(
        self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger
    ) -> ToolOutput:
        try:
            prompt: str = args["prompt"]
            input_character_ranges = args.get("input_character_ranges", [])
            if input_character_ranges is None:
                input_character_ranges = []

            ranges_dict = self._validate_inputs(prompt, input_character_ranges, state)
            prompt = self._format_prompt(prompt, ranges_dict, state)
            response = await self._llm.query(prompt)
            return ToolOutput(
                output=response.output_text_str,
                metadata=response.metadata,
            )
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Retrieve information failed: {error_msg}")
            return ToolOutput(output=error_msg, error=error_msg)


class CourtListenerSearch(Tool):
    name: str = "courtlistener_search"
    description: str = """
    Search and retrieve data from CourtListener, a free database of U.S. federal and state court opinions,
    dockets, judges, citations, and oral arguments.

    Supported modes:
    - "search"   : Keyword search via `/api/rest/v4/search/`; set `search_type` to CourtListener `type`
                   (opinions, dockets/PACER filings, judges, oral argument audio — see enum).
    - "semantic" : Natural-language search across opinions only (`type=o`; API constraint).
    - "hybrid"   : Semantic + required keywords (`type=o` only); quote critical keywords in `query`.
    - "fetch"    : Retrieve a specific opinion, cluster, docket, or judge by CourtListener numeric ID.
    - "citation" : Citation Lookup API (`POST /citation-lookup/`): Eyecite parses text or a volume/reporter/page triad;
                   returns status codes (200/300/404/400/429), normalized citations, and matching clusters.
    - "courts"   : List all courts, or get metadata for a specific court by ID.

    Typical workflow:
      1. Use "search", "semantic", or "citation" (Eyecite citation-lookup API) to find cases and cluster metadata.
      2. Use "fetch" with fetch_type="opinion" and a storage_key to save the full opinion text.
      3. Use retrieve_information to query the saved opinion.
    """.strip()

    parameters: dict[str, Any] = {
        "mode": {
            "type": "string",
            "description": 'The operation to perform. One of: "search", "semantic", "hybrid", "fetch", "citation", "courts".',
            "enum": ["search", "semantic", "hybrid", "fetch", "citation", "courts"],
        },
        # ── SEARCH / SEMANTIC / HYBRID ──────────────────────────────────────────
        "query": {
            "type": "string",
            "description": (
                'Required for modes "search", "semantic", and "hybrid". '
                'For "search": supports boolean operators (AND, OR, NOT) and exact phrases in quotes. '
                'For "semantic": plain natural language (e.g. "prisoners denied medical care"). '
                'For "hybrid": mix natural language with quoted required keywords '
                "(e.g. '\"deliberate indifference\" prison medical care')."
            ),
        },
        "search_type": {
            "type": "string",
            "description": (
                'Only for mode="search": CourtListener search `type` param. '
                '"o" opinions (default), "r" dockets w/ nested docs, "rd" PACER filings, '
                '"d" docket-only PACER, "p" judges, "oa" oral argument audio.'
            ),
            "enum": ["o", "r", "rd", "d", "p", "oa"],
            "default": "o",
        },
        "court": {
            "type": "string",
            "description": '(search/semantic/hybrid only, optional) Filter by court ID. Examples: "scotus", "ca1"–"ca11", "nysd", "cacd". Omit for all courts.',
        },
        "filed_after": {
            "type": "string",
            "description": "(search/semantic/hybrid only, optional) Filter to opinions filed after this date. Format: YYYY-MM-DD.",
        },
        "filed_before": {
            "type": "string",
            "description": "(search/semantic/hybrid only, optional) Filter to opinions filed before this date. Format: YYYY-MM-DD.",
        },
        "page_size": {
            "type": "integer",
            "description": "(search/semantic/hybrid only, optional) Number of results to return. Default: 5, max: 20.",
            "default": 5,
            "minimum": 1,
            "maximum": 20,
        },
        # ── FETCH ────────────────────────────────────────────────────────────────
        "fetch_type": {
            "type": "string",
            "description": (
                "(fetch only) What to retrieve. "
                '"opinion" = full opinion text + metadata. '
                '"cluster" = case-level metadata (citations, judges, sub-opinions). '
                '"docket" = docket metadata (parties, attorneys, assigned judge). '
                '"docket_entries" = individual filings within a docket. '
                '"judge" = judge biographical data.'
            ),
            "enum": ["opinion", "cluster", "docket", "docket_entries", "judge"],
            "default": "opinion",
        },
        "id": {
            "type": "integer",
            "description": "(fetch only) CourtListener numeric ID of the resource to retrieve.",
        },
        "storage_key": {
            "type": "string",
            "description": (
                "(fetch only, optional) If provided, the fetched content is saved to the agent's data storage "
                "under this key instead of being returned directly. Use retrieve_information with {{key}} to query it. "
                "Recommended for long opinions to avoid flooding the context window."
            ),
        },
        # ── CITATION (POST /citation-lookup/) ───────────────────────────────────
        "citation": {
            "type": "string",
            "description": (
                "(citation mode) Single citation or short text passed to Eyecite as `text=` "
                '(e.g. "410 U.S. 113", "576 US 644"). Use `citation_text` instead for long excerpts (max 64k chars). '
                "Mutually exclusive with `citation_text` and with volume/reporter/page."
            ),
        },
        "citation_text": {
            "type": "string",
            "description": (
                "(citation mode) Paragraph or document excerpt to scan for citations (Eyecite); "
                "max 64,000 characters. Mutually exclusive with `citation` and with volume/reporter/page."
            ),
        },
        "volume": {
            "type": "string",
            "description": '(citation mode) Volume for triad lookup with `reporter` and `page` (e.g. "576").',
        },
        "reporter": {
            "type": "string",
            "description": '(citation mode) Reporter for triad lookup (e.g. "U.S.").',
        },
        "page": {
            "type": "string",
            "description": '(citation mode) Page for triad lookup (e.g. "644").',
        },
        # ── COURTS ───────────────────────────────────────────────────────────────
        "court_id": {
            "type": "string",
            "description": '(courts mode only, optional) A specific court ID (e.g. "scotus", "ca9") to fetch metadata for. Omit to list all courts.',
        },
    }
    required: list[str] = ["mode"]

    BASE_URL = "https://www.courtlistener.com/api/rest/v4"

    def __init__(self, courtlistener_api_key: str | None = None):
        super().__init__()
        if not courtlistener_api_key:
            courtlistener_api_key = os.getenv("COURTLISTENER_API_KEY")
        if not courtlistener_api_key:
            raise ValueError("COURTLISTENER_API_KEY is not set")
        # Auth only; omit Content-Type so POST form bodies use application/x-www-form-urlencoded.
        self.headers = {"Authorization": f"Token {courtlistener_api_key}"}

    # ── helpers ──────────────────────────────────────────────────────────────────

    async def _get(self, path: str, params: dict | None = None) -> dict:
        """Single authenticated GET, raises on non-2xx."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.BASE_URL}{path}",
                headers=self.headers,
                params=params,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                resp.raise_for_status()
                return await resp.json()

    @_courtlistener_retry(*COURTLISTENER_RETRY_HTTP_STATUSES)
    async def _post_form(
        self, path: str, data: dict[str, str]
    ) -> list[Any] | dict[str, Any]:
        """POST ``application/x-www-form-urlencoded`` body (CourtListener citation-lookup API)."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.BASE_URL}{path}",
                headers=self.headers,
                data=data,
                timeout=aiohttp.ClientTimeout(total=90),
            ) as resp:
                resp.raise_for_status()
                return await resp.json()

    _CITATION_LOOKUP_TEXT_LIMIT = 64_000

    @staticmethod
    def _trim_cluster_summary(cluster: dict[str, Any]) -> dict[str, Any]:
        sub = cluster.get("sub_opinions") or []
        slim_sub: list[dict[str, Any]] = []
        for o in sub[:20]:
            if isinstance(o, dict):
                slim_sub.append({"id": o.get("id"), "type": o.get("type")})
        return {
            "id": cluster.get("id"),
            "case_name": cluster.get("case_name"),
            "date_filed": cluster.get("date_filed"),
            "absolute_url": cluster.get("absolute_url"),
            "precedential_status": cluster.get("precedential_status"),
            "citations": cluster.get("citations", []),
            "sub_opinions": slim_sub,
        }

    @classmethod
    def _trim_citation_lookup_results(cls, raw: list[Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            clusters_raw = item.get("clusters") or []
            trimmed = [
                cls._trim_cluster_summary(c)
                for c in clusters_raw
                if isinstance(c, dict)
            ]
            out.append(
                {
                    "citation": item.get("citation"),
                    "normalized_citations": item.get("normalized_citations", []),
                    "start_index": item.get("start_index"),
                    "end_index": item.get("end_index"),
                    "status": item.get("status"),
                    "error_message": item.get("error_message", ""),
                    "clusters": trimmed,
                }
            )
        return out

    @staticmethod
    def _trim_search_results(results: list[dict]) -> list[dict]:
        """Return context-friendly subset of search result fields."""
        out = []
        for r in results:
            out.append(
                {
                    "id": r.get("id"),
                    "cluster_id": r.get("cluster_id"),
                    "case_name": r.get("caseName") or r.get("case_name"),
                    "court": r.get("court"),
                    "date_filed": r.get("dateFiled") or r.get("date_filed"),
                    "citation": r.get("citation", []),
                    "status": r.get("status"),
                    "judge": r.get("judge"),
                    "snippet": (r.get("snippet") or "")[:500],
                    "absolute_url": r.get("absolute_url"),
                }
            )
        return out

    # ── search / semantic / hybrid ────────────────────────────────────────────────

    @_courtlistener_retry(*COURTLISTENER_RETRY_HTTP_STATUSES)
    async def _search(
        self,
        query: str,
        semantic: bool = False,
        search_type: str = "o",
        court: str | None = None,
        filed_after: str | None = None,
        filed_before: str | None = None,
        page_size: int = 5,
    ) -> list[dict]:
        params: dict[str, Any] = {
            "q": query,
            "type": search_type,
            "order_by": "score desc",
            "page_size": min(page_size, 20),
        }
        if semantic:
            params["semantic"] = "true"
        if court:
            params["court"] = court
        if filed_after:
            params["filed_after"] = filed_after
        if filed_before:
            params["filed_before"] = filed_before

        data = await self._get("/search/", params)
        return self._trim_search_results(data.get("results", []))

    # ── fetch ─────────────────────────────────────────────────────────────────────

    @_courtlistener_retry(*COURTLISTENER_RETRY_HTTP_STATUSES)
    async def _fetch(self, fetch_type: str, id: int) -> dict:
        ep_map = {
            "opinion": f"/opinions/{id}/",
            "cluster": f"/clusters/{id}/",
            "docket": f"/dockets/{id}/",
            "docket_entries": "/docket-entries/",
            "judge": f"/people/{id}/",
        }
        params = {"docket": id} if fetch_type == "docket_entries" else None
        data = await self._get(ep_map[fetch_type], params)

        if fetch_type == "opinion":
            return {
                "id": data.get("id"),
                "cluster_id": data.get("cluster_id"),
                "author": data.get("author_str"),
                "date_created": data.get("date_created"),
                "type": data.get("type"),
                "text": (
                    data.get("plain_text")
                    or _strip_html(data.get("html_with_citations", ""))
                    or _strip_html(data.get("html", ""))
                    or _strip_html(data.get("xml_harvard", ""))
                    or "(no text available)"
                ),
            }
        if fetch_type == "cluster":
            return {
                "id": data.get("id"),
                "case_name": data.get("case_name"),
                "date_filed": data.get("date_filed"),
                "docket": data.get("docket"),
                "citations": data.get("citations", []),
                "judges": data.get("judges"),
                "precedential_status": data.get("precedential_status"),
                "syllabus": data.get("syllabus", ""),
                "sub_opinions": data.get("sub_opinions", []),
                "absolute_url": data.get("absolute_url"),
            }
        if fetch_type == "docket":
            return {
                "id": data.get("id"),
                "case_name": data.get("case_name"),
                "court": data.get("court_id"),
                "date_filed": data.get("date_filed"),
                "date_terminated": data.get("date_terminated"),
                "assigned_to": data.get("assigned_to_str"),
                "referred_to": data.get("referred_to_str"),
                "cause": data.get("cause"),
                "nature_of_suit": data.get("nature_of_suit"),
                "absolute_url": data.get("absolute_url"),
            }
        if fetch_type == "docket_entries":
            entries = data.get("results", [])
            return {
                "count": data.get("count"),
                "entries": [
                    {
                        "id": e.get("id"),
                        "date_filed": e.get("date_filed"),
                        "entry_number": e.get("entry_number"),
                        "description": (e.get("description") or "")[:300],
                        "recap_documents": [
                            {
                                "id": d.get("id"),
                                "description": (d.get("description") or "")[:200],
                            }
                            for d in e.get("recap_documents", [])
                        ],
                    }
                    for e in entries
                ],
            }
        if fetch_type == "judge":
            return {
                "id": data.get("id"),
                "name_full": data.get("name_full"),
                "name_first": data.get("name_first"),
                "name_last": data.get("name_last"),
                "date_dob": data.get("date_dob"),
                "date_dod": data.get("date_dod"),
                "gender": data.get("gender"),
                "race": data.get("race", []),
                "positions": data.get("positions", []),
                "educations": data.get("educations", []),
                "aba_ratings": data.get("aba_ratings", []),
                "political_affiliations": data.get("political_affiliations", []),
            }
        return data

    # ── citation lookup (Eyecite POST /citation-lookup/) ──────────────────────────

    async def _citation_lookup_request(
        self, form: dict[str, str]
    ) -> list[dict[str, Any]]:
        raw = await self._post_form("/citation-lookup/", form)
        if not isinstance(raw, list):
            raise ValueError(
                f"Unexpected citation-lookup response: {type(raw).__name__}"
            )
        return self._trim_citation_lookup_results(raw)

    # ── courts ────────────────────────────────────────────────────────────────────

    @_courtlistener_retry(*COURTLISTENER_RETRY_HTTP_STATUSES)
    async def _courts(self, court_id: str | None = None) -> dict | list:
        if court_id:
            return await self._get(f"/courts/{court_id}/")
        data = await self._get("/courts/", {"page_size": 500})
        return [
            {
                "id": c.get("id"),
                "full_name": c.get("full_name"),
                "short_name": c.get("short_name"),
                "jurisdiction": c.get("jurisdiction"),
                "in_use": c.get("in_use"),
            }
            for c in data.get("results", [])
        ]

    # ── storage helper (mirrors ParseHtmlPage) ────────────────────────────────────

    def _save_to_storage(self, content: Any, key: str, data_storage: dict) -> str:
        text = json.dumps(content, indent=2, default=str)
        warn = (
            f"WARNING: Key '{key}' already exists and will be overwritten.\n"
            if key in data_storage
            else ""
        )
        data_storage[key] = text
        keys_list = "\n".join(data_storage.keys())
        return f"{warn}SUCCESS: Content saved under key: '{key}'.\nData storage keys:\n{keys_list}"

    # ── main dispatch ─────────────────────────────────────────────────────────────

    async def execute(
        self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger
    ) -> ToolOutput:
        try:
            result = await self.call_tool(
                arguments=args,
                data_storage=state,
                llm=None,  # CourtListener API does not use LLM
                logger=logger,
            )
            return ToolOutput(output=json.dumps(result, default=str))
        except RetryExhaustedError:
            raise
        except aiohttp.ClientResponseError as e:
            if e.status in COURTLISTENER_RETRY_HTTP_STATUSES:
                raise RetryExhaustedError(
                    f"CourtListener API retry attempts exhausted for HTTP {e.status}: {e}"
                ) from e
            error_msg = str(e)
            logger.warning(f"CourtListener search failed: {error_msg}")
            return ToolOutput(output=error_msg, error=error_msg)
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"CourtListener search failed: {error_msg}")
            return ToolOutput(output=error_msg, error=error_msg)

    async def call_tool(
        self,
        arguments: dict[str, Any],
        data_storage: dict[str, Any],
        llm: LLM | None = None,
        logger: logging.Logger | None = None,
    ) -> dict[str, Any]:
        mode = arguments.get("mode")

        # ── search (keyword) ──
        if mode == "search":
            query = arguments.get("query")
            if not query:
                raise ValueError('"query" is required for mode="search".')
            results = await self._search(
                query=query,
                semantic=False,
                search_type=arguments.get("search_type", "o"),
                court=arguments.get("court"),
                filed_after=arguments.get("filed_after"),
                filed_before=arguments.get("filed_before"),
                page_size=arguments.get("page_size", 5),
            )
            return {"results": results}

        # ── semantic search ──
        elif mode == "semantic":
            query = arguments.get("query")
            if not query:
                raise ValueError('"query" is required for mode="semantic".')
            results = await self._search(
                query=query,
                semantic=True,
                search_type="o",  # semantic only supported for opinions
                court=arguments.get("court"),
                filed_after=arguments.get("filed_after"),
                filed_before=arguments.get("filed_before"),
                page_size=arguments.get("page_size", 5),
            )
            return {"results": results}

        # ── hybrid search ──
        elif mode == "hybrid":
            query = arguments.get("query")
            if not query:
                raise ValueError('"query" is required for mode="hybrid".')
            # Hybrid = semantic=true with quoted keywords in the query string
            results = await self._search(
                query=query,
                semantic=True,
                search_type="o",
                court=arguments.get("court"),
                filed_after=arguments.get("filed_after"),
                filed_before=arguments.get("filed_before"),
                page_size=arguments.get("page_size", 5),
            )
            return {"results": results}

        # ── fetch ──
        elif mode == "fetch":
            resource_id = arguments.get("id")
            if resource_id is None:
                raise ValueError('"id" is required for mode="fetch".')
            fetch_type = arguments.get("fetch_type", "opinion")
            result = await self._fetch(fetch_type=fetch_type, id=resource_id)
            storage_key = arguments.get("storage_key")
            if storage_key:
                msg = self._save_to_storage(result, storage_key, data_storage)
                return {"message": msg}
            return {"data": result}

        # ── citation lookup (POST /citation-lookup/) ──
        elif mode == "citation":
            vol = arguments.get("volume")
            rep = arguments.get("reporter")
            page = arguments.get("page")
            triad_parts = [vol, rep, page]
            triad_filled = sum(
                1 for x in triad_parts if x is not None and str(x).strip()
            )
            if triad_filled in (1, 2):
                raise ValueError(
                    'For triad lookup, provide all of "volume", "reporter", and "page".'
                )
            triad = triad_filled == 3

            cit = (arguments.get("citation") or "").strip()
            blob = (arguments.get("citation_text") or "").strip()

            if triad and (cit or blob):
                raise ValueError(
                    'Use either volume/reporter/page OR citation/citation_text for mode="citation".'
                )

            if triad:
                form = {
                    "volume": str(vol).strip(),
                    "reporter": str(rep).strip(),
                    "page": str(page).strip(),
                }
            elif blob:
                if len(blob) > self._CITATION_LOOKUP_TEXT_LIMIT:
                    raise ValueError(
                        f"citation_text exceeds CourtListener limit of {self._CITATION_LOOKUP_TEXT_LIMIT} characters."
                    )
                form = {"text": blob}
            elif cit:
                if len(cit) > self._CITATION_LOOKUP_TEXT_LIMIT:
                    raise ValueError(
                        f"citation exceeds CourtListener limit of {self._CITATION_LOOKUP_TEXT_LIMIT} characters."
                    )
                form = {"text": cit}
            else:
                raise ValueError(
                    'Provide "citation", "citation_text", or volume/reporter/page for mode="citation".'
                )

            results = await self._citation_lookup_request(form)
            return {"results": results}

        # ── courts ──
        elif mode == "courts":
            data = await self._courts(court_id=arguments.get("court_id"))
            if isinstance(data, list):
                return {"courts": data}
            return {"court": data}

        else:
            raise ValueError(
                f'Unknown mode "{mode}". Must be one of: search, semantic, hybrid, fetch, citation, courts.'
            )


def _strip_html(html: str) -> str:
    """Lightweight HTML/XML stripper for opinion text fallback."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n").strip()
