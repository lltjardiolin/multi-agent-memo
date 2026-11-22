# Week 18 Research Memo Generator — learner starter

This starter provides the promised fictional corpus, MCP server, five fixed topics, a single-agent baseline and an evaluation runner, plus a LangGraph supervisor template. It is not a completed solution. Implement the TODOs in `team.py`; use the same corpus, model, rubric, and topics for both systems. No real market data is included.

## Setup

Use Python 3.11 or later. In a terminal at this folder:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python check_server.py
```

The last command launches the MCP server via stdio, lists its resources and tools, reads both resources, and searches for `solar`. It requires no API key. Ensure Vocareum allows package installation and subprocesses; ask your course team if these are unavailable.

## Using the OpenAI access provided in Vocareum

Your programme provides OpenAI API access through Vocareum. You do not need to purchase a personal OpenAI subscription or API key for this assignment. Use the course-provided access and contact the course team if the key, model, endpoint or available quota is unclear. The exact environment-variable names and supported models depend on the lab configuration.

The supplied `baseline.py` reads three environment variables: `MODEL_API_KEY`, `MODEL_BASE_URL` and `MODEL_NAME`. If your course instructions confirm that the lab exposes its key as `OPENAI_API_KEY`, you can set the baseline's key variable from it without displaying the key:

```bash
export MODEL_API_KEY="$OPENAI_API_KEY"
```

If the key is available under a different variable name, use the name specified by the course team instead. If Vocareum already sets `MODEL_API_KEY`, leave it unchanged. If the key is not exposed as an environment variable, follow the lab's secure access instructions or ask the course team how to configure the baseline. Do not paste the key into code, a notebook, screenshots or a submitted `.env` file, and do not print it in the terminal.

Set the endpoint and model using the values confirmed for your lab:

```bash
export MODEL_BASE_URL='COURSE_CONFIRMED_ENDPOINT'
export MODEL_NAME='COURSE_CONFIRMED_MODEL_NAME'
```

Replace both placeholders with the values provided by the course team. The standard OpenAI API endpoint is `https://api.openai.com/v1`; use it only if the course confirms that the lab connects directly to OpenAI. A Vocareum proxy may require a different endpoint. Select an enabled model, rather than assuming any particular model is available. If `MODEL_BASE_URL` or `MODEL_NAME` is already set correctly in the lab, do not overwrite it.

To check whether all three settings are present without exposing their values, run:

```bash
python -c 'import os; names=("MODEL_API_KEY", "MODEL_BASE_URL", "MODEL_NAME"); [print(n + ": " + ("set" if os.getenv(n) else "not set")) for n in names]'
```

This checks presence only; it does not verify that the key, endpoint, model or quota works. Run `python evaluate.py --output results/evaluation.json` to exercise the baseline after setup; check the output file for each run's `status`. The team entries will report errors until `team.run_topic` is implemented. If configuration or model access fails, contact the course team; do not create a personal paid account. The baseline sends the fictional corpus to the configured provider. Use the same approved model and settings for both approaches and report any differences.

## Files and tasks

* `data/market_corpus.md` contains 12 numbered fictional paragraphs. `data/corpus.csv` contains exactly the same paragraphs with stable citation IDs; `data/glossary.md` defines terms.
* `mcp_server.py` exposes the CSV and glossary as MCP resources and `search_corpus` as a tool. `check_server.py` demonstrates a real stdio client and tool call. Adapt the client pattern for the Researcher.
* `topics.json` contains five fixed topic IDs and prompts. IDs are reproducibility labels, not random-number seeds. If your model supports a random seed, document its setting separately.
* `team.py` contains state and routing scaffolding. Implement the supervisor, three workers, typed handoffs, MCP retrieval, state persistence/resume, memory compaction and trace collection. The supplied template intentionally raises `NotImplementedError`.
* `baseline.py` is a provided single-agent comparator. `evaluate.py` runs both systems on all five topics and writes JSON. It records failures honestly; an incomplete team yields error records, not successful results.

Run `python evaluate.py --output results/evaluation.json` after configuring the model and implementing `team.run_topic`. It can be run earlier to check integration, but the team entries will report errors. See `EVALUATION.md` for return format, metrics and limitations. Generated results are your own; no precomputed successful result is provided.

## Deliverables

Submit the complete implementation and generated evidence as required by the LMS. The original assignment specifies a code ZIP and PDF report. Provide a concise project README (at most one page), and exclude credentials, `.venv`, caches and unrelated files. Do not share this starter as a solution.
