import anyio
import click
import httpx
import asyncio
from langchain_openai import ChatOpenAI
from browser_use import Agent
from browser_use.browser.browser import Browser, BrowserConfig
import mcp.types as types
from mcp.server.lowlevel import Server
from dotenv import load_dotenv
import json
import logging
from browser_use.browser.context import BrowserContextConfig, BrowserContext

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Browser context configuration
config = BrowserContextConfig(
    wait_for_network_idle_page_load_time=0.6,
    maximum_wait_page_load_time=1.2,
    minimum_wait_page_load_time=0.2,
    browser_window_size={"width": 1280, "height": 1100},
    locale="en-US",
    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36",
    highlight_elements=True,
    viewport_expansion=0,
)

# Initialize browser and context
browser = Browser()
context = BrowserContext(browser=browser, config=config)

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

# Flag to track browser context health
browser_context_healthy = True


async def reset_browser_context():
    """Reset the browser context to a clean state."""
    global context, browser, browser_context_healthy

    logger.info("Resetting browser context")
    try:
        # Try to close the existing context
        try:
            await context.close()
        except Exception as e:
            logger.warning(f"Error closing browser context: {str(e)}")

        # Create a new context
        context = BrowserContext(browser=browser, config=config)
        browser_context_healthy = True
        logger.info("Browser context reset successfully")
    except Exception as e:
        logger.error(f"Failed to reset browser context: {str(e)}")
        browser_context_healthy = False
        # If we can't reset the context, try to reset the browser
        try:
            await browser.close()
            browser = Browser()
            context = BrowserContext(browser=browser, config=config)
            browser_context_healthy = True
            logger.info("Browser reset successfully")
        except Exception as e:
            logger.error(f"Failed to reset browser: {str(e)}")
            browser_context_healthy = False


async def check_browser_health():
    """Check if the browser context is healthy."""
    global browser_context_healthy

    if not browser_context_healthy:
        await reset_browser_context()
        return browser_context_healthy

    try:
        # Simple health check - try to get the current page
        await context.get_current_page()
        return True
    except Exception as e:
        logger.warning(f"Browser health check failed: {str(e)}")
        browser_context_healthy = False
        await reset_browser_context()
        return browser_context_healthy


async def browser_use(
    url: str,
    action: str,
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Perform a browser action and return the results."""
    global browser_context_healthy

    headers = {
        "User-Agent": "browser-use (github.com/co-browser/browser-use-mcp-server)",
    }

    # Check browser health before proceeding
    if not await check_browser_health():
        return [
            types.TextContent(
                type="text",
                text=json.dumps(
                    {
                        "final_result": "Browser context is unhealthy and could not be reset",
                        "success": False,
                        "has_errors": True,
                        "errors": [
                            "Browser context is unhealthy and could not be reset"
                        ],
                        "urls_visited": [],
                        "actions_performed": [],
                        "extracted_content": [],
                        "steps_taken": 0,
                    },
                    indent=2,
                ),
            )
        ]

    try:
        # Use the existing browser context
        agent = Agent(task=action, llm=llm, browser_context=context)
        ret = await agent.run(max_steps=10)

        # Get the final result
        final_result = ret.final_result()

        # Check if we have a valid result
        if final_result and hasattr(final_result, "raise_for_status"):
            final_result.raise_for_status()
            result_text = str(final_result.text)
        else:
            result_text = (
                str(final_result) if final_result else "No final result available"
            )

        # Gather essential information from the agent history
        is_successful = ret.is_successful()
        has_errors = ret.has_errors()
        errors = ret.errors()
        urls_visited = ret.urls()
        action_names = ret.action_names()
        extracted_content = ret.extracted_content()
        steps_taken = ret.number_of_steps()

        # Create a focused response with the most relevant information for an LLM
        response_data = {
            "final_result": result_text,
            "success": is_successful,
            "has_errors": has_errors,
            "errors": [str(err) for err in errors if err],
            "urls_visited": [str(url) for url in urls_visited if url],
            "actions_performed": action_names,
            "extracted_content": extracted_content,
            "steps_taken": steps_taken,
        }

        # Convert to JSON string
        response_json = json.dumps(response_data, indent=2)

        return [types.TextContent(type="text", text=response_json)]

    except Exception as e:
        logger.error(f"Error in browser_use: {str(e)}")
        import traceback

        tb = traceback.format_exc()

        # Mark the browser context as unhealthy
        browser_context_healthy = False

        # Return error information
        error_message = {
            "final_result": f"Error: {str(e)}",
            "success": False,
            "has_errors": True,
            "errors": [str(e), tb],
            "urls_visited": [],
            "actions_performed": [],
            "extracted_content": [],
            "steps_taken": 0,
        }

        return [
            types.TextContent(type="text", text=json.dumps(error_message, indent=2))
        ]

    finally:
        # Always try to reset the browser context to a clean state after use
        # This helps prevent issues with subsequent requests
        try:
            # For now, we'll just navigate to about:blank to reset the page state
            # This is less resource-intensive than creating a new context each time
            current_page = await context.get_current_page()
            await current_page.goto("about:blank")
        except Exception as e:
            logger.warning(f"Error resetting page state: {str(e)}")
            # If we can't reset the page state, mark the context as unhealthy
            browser_context_healthy = False


@click.command()
@click.option("--port", default=8000, help="Port to listen on for SSE")
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    help="Transport type",
)
@click.option(
    "--timeout",
    default=120,
    help="Timeout in seconds for tool execution",
)
def main(port: int, transport: str, timeout: int) -> int:
    load_dotenv()
    app = Server("browser_use")

    @app.call_tool()
    async def call_tool(
        name: str, arguments: dict
    ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        global browser_context_healthy

        if name != "browser_use":
            raise ValueError(f"Unknown tool: {name}")
        if "url" not in arguments:
            raise ValueError("Missing required argument 'url'")
        if "action" not in arguments:
            raise ValueError("Missing required argument 'action'")

        # Create a task for the browser_use function
        browser_task = None

        try:
            # Create a task for the browser_use function
            browser_task = asyncio.create_task(
                browser_use(arguments["url"], arguments["action"])
            )

            # Wait for the task to complete with a timeout
            try:
                result = await asyncio.wait_for(browser_task, timeout=timeout)
                return result
            except asyncio.TimeoutError:
                # Cancel the task if it times out
                if browser_task and not browser_task.done():
                    browser_task.cancel()

                # Mark the browser context as unhealthy
                browser_context_healthy = False

                # Schedule a reset of the browser context
                asyncio.create_task(reset_browser_context())

                # Return a meaningful error message if the operation times out
                error_message = {
                    "final_result": "Operation timed out",
                    "success": False,
                    "has_errors": True,
                    "errors": [
                        f"The operation exceeded the {timeout} second timeout limit"
                    ],
                    "urls_visited": [],
                    "actions_performed": [],
                    "extracted_content": [],
                    "steps_taken": 0,
                }
                return [
                    types.TextContent(
                        type="text", text=json.dumps(error_message, indent=2)
                    )
                ]
        except Exception as e:
            # Cancel the task if it's still running
            if browser_task and not browser_task.done():
                browser_task.cancel()

            # Handle other exceptions gracefully
            import traceback

            tb = traceback.format_exc()

            # Mark the browser context as unhealthy
            browser_context_healthy = False

            # Schedule a reset of the browser context
            asyncio.create_task(reset_browser_context())

            error_message = {
                "final_result": f"Error: {str(e)}",
                "success": False,
                "has_errors": True,
                "errors": [str(e), tb],
                "urls_visited": [],
                "actions_performed": [],
                "extracted_content": [],
                "steps_taken": 0,
            }
            return [
                types.TextContent(type="text", text=json.dumps(error_message, indent=2))
            ]

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="browser_use",
                description="takes a prompt representing an action to perform in the browser and returns detailed information about the execution",
                inputSchema={
                    "type": "object",
                    "required": ["url", "action"],
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL to navigate to",
                        },
                        "action": {
                            "type": "string",
                            "description": "Action to perform in the browser",
                        },
                    },
                },
                outputSchema={
                    "type": "object",
                    "properties": {
                        "final_result": {
                            "type": "string",
                            "description": "The final result of the browser action",
                        },
                        "success": {
                            "type": "boolean",
                            "description": "Whether the action was successful",
                        },
                        "has_errors": {
                            "type": "boolean",
                            "description": "Whether any errors occurred during execution",
                        },
                        "errors": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of errors that occurred during execution",
                        },
                        "urls_visited": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of URLs visited during execution",
                        },
                        "actions_performed": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of actions performed during execution",
                        },
                        "extracted_content": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Content extracted during execution",
                        },
                        "steps_taken": {
                            "type": "integer",
                            "description": "Number of steps taken during execution",
                        },
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
            try:
                async with sse.connect_sse(
                    request.scope, request.receive, request._send
                ) as streams:
                    await app.run(
                        streams[0], streams[1], app.create_initialization_options()
                    )
            except Exception as e:
                logger.error(f"Error in handle_sse: {str(e)}")
                # Ensure browser context is reset if there's an error
                asyncio.create_task(reset_browser_context())
                raise

        starlette_app = Starlette(
            debug=True,
            routes=[
                Route("/sse", endpoint=handle_sse),
                Mount("/messages/", app=sse.handle_post_message),
            ],
        )

        import uvicorn

        # Add a startup event to initialize the browser
        @starlette_app.on_event("startup")
        async def startup_event():
            global browser, context, browser_context_healthy
            try:
                # Ensure browser and context are initialized
                if not browser_context_healthy:
                    await reset_browser_context()
            except Exception as e:
                logger.error(f"Error during startup: {str(e)}")

        # Add a shutdown event to clean up resources
        @starlette_app.on_event("shutdown")
        async def shutdown_event():
            global browser, context
            try:
                # Close the browser and context
                await context.close()
                await browser.close()
            except Exception as e:
                logger.error(f"Error during shutdown: {str(e)}")

        uvicorn.run(starlette_app, host="0.0.0.0", port=port)
    else:
        from mcp.server.stdio import stdio_server

        async def arun():
            try:
                # Ensure browser context is healthy before starting
                await check_browser_health()

                async with stdio_server() as streams:
                    await app.run(
                        streams[0], streams[1], app.create_initialization_options()
                    )
            except Exception as e:
                logger.error(f"Error in arun: {str(e)}")
                # Ensure browser context is reset if there's an error
                await reset_browser_context()
            finally:
                # Clean up resources
                try:
                    await context.close()
                    await browser.close()
                except Exception as e:
                    logger.error(f"Error cleaning up resources: {str(e)}")

        anyio.run(arun)

    return 0
