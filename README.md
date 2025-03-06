# ➡️ browser-use mcp server 

[browser-use](https://github.com/browser-use/browser-use) MCP Server with SSE transport

### requirements

- uv

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### quickstart

```
uv sync
uv pip install playwright
uv run playwright install --with-deps --no-shell chromium
uv run server --transport sse --port 8000 &
uv run client
```

### supported clients

- cursor.ai
- claude desktop
- claude code
- windsurf

### usage

after running the server, add http://localhost:8000/sse to your client, then try asking your LLM the following:

```open https://news.ycombinator.com and return the top ranked article```

### help

for issues or interest reach out @ https://cobrowser.xyz

