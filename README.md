# browser-use mcp server

browser-use MCP Server with SSE transport

### requirements

- uv

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### quickstart

```
uv pip install playwright
uv run playwright install --with-deps --no-shell chromium
uv sync
uv run server --transport sse --port 8000 &
uv run client
```

### supported clients

- cursor.ai
- claude desktop
- claude code
- windsurf

### usage

after running the server, add http://localhost:8000/sse to your client UI, or in a mcp.json file:

```json
{
  "mcpServers": {
    "browser-use-mcp=server": {
      "url": "http://localhost:8000/sse"
    }
  }
}
```
#### cursor
- ./.cursor/mcp.json
#### windsurf
- ~/.codeium/windsurf/mcp_config.json
#### claude
macos
- ~/Library/Application Support/Claude/claude_desktop_config.json
windows
- %APPDATA%\Claude\claude_desktop_config.json


then try asking your LLM the following:

```open https://news.ycombinator.com and return the top ranked article```

