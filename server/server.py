import anyio
import click
import httpx
import asyncio
from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser
import mcp.types as types
from mcp.server.lowlevel import Server
from dotenv import load_dotenv
from browser_use.browser.context import BrowserContextConfig, BrowserContext

config = BrowserContextConfig(
    wait_for_network_idle_page_load_time=0.6,
    maximum_wait_page_load_time=1.2,
    minimum_wait_page_load_time=0.2,
    browser_window_size={'width': 1280, 'height': 1100},
    locale='en-US',
    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36',
    highlight_elements=True,
    viewport_expansion=500,
)

browser = Browser()
context = BrowserContext(browser=browser, config=config)

llm = ChatOpenAI(model='gpt-4o', temperature=0.0)
planner_llm = ChatOpenAI(
	model='o3-mini',
)

async def browser_use(
    url: str,
    action: str,
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    headers = {
        "User-Agent": "browser-use (github.com/co-browser/browser-use-mcp-server)",
    }
    agent = Agent(task=action, llm=llm, browser_context=context)
    ret = await agent.run()
    response = ret.final_result()
    return [types.TextContent(type="text", text=response)]


@click.command()
@click.option("--port", default=8000, help="Port to listen on for SSE")
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    help="Transport type",
)
def main(port: int, transport: str) -> int:
    load_dotenv()
    app = Server("browser_use")
    @app.call_tool()
    async def call_tool(
        name: str, arguments: dict
    ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        if name != "browser_use":
            raise ValueError(f"Unknown tool: {name}")
        if "url" not in arguments:
            raise ValueError("Missing required argument 'url'")
        if "action" not in arguments:
            raise ValueError("Missing required argument 'action'")
        return await browser_use(arguments["url"], arguments["action"])

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="browser_use",
                description="takes a prompt representing an action to perform in the browser",
                inputSchema={
                    "type": "object",
                    "required": ["url", "action"],
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL to fetch",
                        },
                        "action": {
                            "type": "string",
                            "description": "Action to perform",
                        }
                    },
                },
            )
        ]

    if transport == "sse":
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Mount, Route

        sse = SseServerTransport("/messages/")

        async def handle_sse(request):
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await app.run(
                    streams[0], streams[1], app.create_initialization_options()
                )

        starlette_app = Starlette(
            debug=True,
            routes=[
                Route("/sse", endpoint=handle_sse),
                Mount("/messages/", app=sse.handle_post_message),
            ],
        )

        import uvicorn

        uvicorn.run(starlette_app, host="0.0.0.0", port=port)
    else:
        from mcp.server.stdio import stdio_server

        async def arun():
            async with stdio_server() as streams:
                await app.run(
                    streams[0], streams[1], app.create_initialization_options()
                )

        anyio.run(arun)

    return 0