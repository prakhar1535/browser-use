import anyio
import click
import httpx
import asyncio
from langchain_openai import ChatOpenAI
from browser_use import Agent
import mcp.types as types
from mcp.server.lowlevel import Server
from dotenv import load_dotenv

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
    async with httpx.AsyncClient(follow_redirects=True, headers=headers) as client:
        agent = Agent(task=action, llm=llm, planner_llm=planner_llm, use_vision_for_planner=False, planner_interval=1)
        ret = await agent.run()
        response = ret
        response.raise_for_status()
        return [types.TextContent(type="text", text=response.text)]


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
        return await browser_use(arguments["url", "action"])

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