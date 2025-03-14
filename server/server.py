"""
Browser Use MCP Server

This module implements an MCP (Model-Control-Protocol) server for browser automation
using the browser_use library. It provides functionality to interact with a browser instance
via an async task queue, allowing for long-running browser tasks to be executed asynchronously
while providing status updates and results.

The server supports Server-Sent Events (SSE) for web-based interfaces.
"""

import os
import click
import asyncio
import uuid
from datetime import datetime
from dotenv import load_dotenv
import logging
import json
import traceback

# Import from browser-use library
from browser_use.browser.context import BrowserContextConfig, BrowserContext
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use import Agent

# Import MCP server components
from mcp.server.lowlevel import Server
import mcp.types as types

# Import LLM provider
from langchain_openai import ChatOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Task storage for async operations
task_store = {}

# Flag to track browser context health
browser_context_healthy = True

# Store global browser context and configuration
browser = None
context = None
config = None


async def reset_browser_context():
    """
    Reset the browser context to a clean state.

    This function attempts to close the existing context and create a new one.
    If that fails, it tries to recreate the entire browser instance.
    """
    global context, browser, browser_context_healthy, config

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

            # Recreate browser with same configuration
            browser_config = BrowserConfig(
                extra_chromium_args=[
                    "--no-sandbox",
                    "--disable-gpu",
                    "--disable-software-rasterizer",
                    "--disable-dev-shm-usage",
                    "--remote-debugging-port=9222",
                ],
            )

            # Only set chrome_instance_path if we have a path from environment or command line
            chrome_path = os.environ.get("CHROME_PATH")
            if chrome_path:
                browser_config.chrome_instance_path = chrome_path

            browser = Browser(config=browser_config)
            context = BrowserContext(browser=browser, config=config)
            browser_context_healthy = True
            logger.info("Browser reset successfully")
        except Exception as e:
            logger.error(f"Failed to reset browser: {str(e)}")
            browser_context_healthy = False


async def check_browser_health():
    """
    Check if the browser context is healthy by attempting to access the current page.

    If the context is unhealthy, attempts to reset it.

    Returns:
        bool: True if the browser context is healthy, False otherwise.
    """
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


async def run_browser_task_async(task_id, url, action, llm):
    """
    Run a browser task asynchronously and store the result.

    This function executes a browser automation task with the given URL and action,
    and updates the task store with progress and results.

    Args:
        task_id (str): Unique identifier for the task
        url (str): URL to navigate to
        action (str): Action to perform after navigation
        llm: Language model to use for browser agent
    """
    try:
        # Update task status to running
        task_store[task_id]["status"] = "running"
        task_store[task_id]["start_time"] = datetime.now().isoformat()
        task_store[task_id]["progress"] = {
            "current_step": 0,
            "total_steps": 0,
            "steps": [],
        }

        # Reset browser context to ensure a clean state
        await reset_browser_context()

        # Check browser health
        if not await check_browser_health():
            task_store[task_id]["status"] = "failed"
            task_store[task_id]["end_time"] = datetime.now().isoformat()
            task_store[task_id]["error"] = (
                "Browser context is unhealthy and could not be reset"
            )
            return

        # Define step callback function with the correct signature
        async def step_callback(browser_state, agent_output, step_number):
            # Update progress in task store
            task_store[task_id]["progress"]["current_step"] = step_number
            task_store[task_id]["progress"]["total_steps"] = max(
                task_store[task_id]["progress"]["total_steps"], step_number
            )

            # Add step info with minimal details
            step_info = {"step": step_number, "time": datetime.now().isoformat()}

            # Add goal if available
            if agent_output and hasattr(agent_output, "current_state"):
                if hasattr(agent_output.current_state, "next_goal"):
                    step_info["goal"] = agent_output.current_state.next_goal

            # Add to progress steps
            task_store[task_id]["progress"]["steps"].append(step_info)

            # Log progress
            logger.info(f"Task {task_id}: Step {step_number} completed")

        # Define done callback function with the correct signature
        async def done_callback(history):
            # Log completion
            logger.info(f"Task {task_id}: Completed with {len(history.history)} steps")

            # Add final step
            current_step = task_store[task_id]["progress"]["current_step"] + 1
            task_store[task_id]["progress"]["steps"].append(
                {
                    "step": current_step,
                    "time": datetime.now().isoformat(),
                    "status": "completed",
                }
            )

        # Use the existing browser context with callbacks
        agent = Agent(
            task=f"First, navigate to {url}. Then, {action}",
            llm=llm,
            browser_context=context,
            register_new_step_callback=step_callback,
            register_done_callback=done_callback,
        )

        # Run the agent with a reasonable step limit
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

        # Create a focused response with the most relevant information
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

        # Store the result
        task_store[task_id]["status"] = "completed"
        task_store[task_id]["end_time"] = datetime.now().isoformat()
        task_store[task_id]["result"] = response_data

    except Exception as e:
        logger.error(f"Error in async browser task: {str(e)}")
        tb = traceback.format_exc()

        # Mark the browser context as unhealthy
        global browser_context_healthy
        browser_context_healthy = False

        # Store the error
        task_store[task_id]["status"] = "failed"
        task_store[task_id]["end_time"] = datetime.now().isoformat()
        task_store[task_id]["error"] = str(e)
        task_store[task_id]["traceback"] = tb

    finally:
        # Always try to reset the browser context to a clean state after use
        try:
            current_page = await context.get_current_page()
            await current_page.goto("about:blank")
        except Exception as e:
            logger.warning(f"Error resetting page state: {str(e)}")
            browser_context_healthy = False


async def cleanup_old_tasks():
    """
    Periodically clean up old completed tasks to prevent memory leaks.

    This function runs continuously in the background, removing tasks that have been
    completed or failed for more than 1 hour to conserve memory.
    """
    while True:
        try:
            # Sleep first to avoid cleaning up tasks too early
            await asyncio.sleep(3600)  # Run cleanup every hour

            current_time = datetime.now()
            tasks_to_remove = []

            # Find completed tasks older than 1 hour
            for task_id, task_data in task_store.items():
                if (
                    task_data["status"] in ["completed", "failed"]
                    and "end_time" in task_data
                ):
                    end_time = datetime.fromisoformat(task_data["end_time"])
                    hours_elapsed = (current_time - end_time).total_seconds() / 3600

                    if hours_elapsed > 1:  # Remove tasks older than 1 hour
                        tasks_to_remove.append(task_id)

            # Remove old tasks
            for task_id in tasks_to_remove:
                del task_store[task_id]

            if tasks_to_remove:
                logger.info(f"Cleaned up {len(tasks_to_remove)} old tasks")

        except Exception as e:
            logger.error(f"Error in task cleanup: {str(e)}")


def create_mcp_server(llm, task_expiry_minutes=60):
    """
    Create and configure an MCP server for browser interaction.

    Args:
        llm: The language model to use for browser agent
        task_expiry_minutes (int): Minutes after which tasks are considered expired

    Returns:
        Server: Configured MCP server instance
    """
    # Create MCP server instance
    app = Server("browser_use")

    @app.call_tool()
    async def call_tool(
        name: str, arguments: dict
    ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        global browser_context_healthy

        # Handle browser_use tool
        if name == "browser_use":
            # Check required arguments
            if "url" not in arguments:
                raise ValueError("Missing required argument 'url'")
            if "action" not in arguments:
                raise ValueError("Missing required argument 'action'")

            # Generate a task ID
            task_id = str(uuid.uuid4())

            # Initialize task in store
            task_store[task_id] = {
                "id": task_id,
                "status": "pending",
                "url": arguments["url"],
                "action": arguments["action"],
                "created_at": datetime.now().isoformat(),
            }

            # Start task in background
            asyncio.create_task(
                run_browser_task_async(
                    task_id, arguments["url"], arguments["action"], llm
                )
            )

            # Estimate task duration
            estimated_seconds = 60  # Default estimate of 60 seconds

            # Return task ID immediately with explicit sleep instruction
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "task_id": task_id,
                            "status": "pending",
                            "message": f"Browser task started. Please wait for {estimated_seconds} seconds, then check the result using browser_get_result or the resource URI. Always wait exactly 5 seconds between status checks.",
                            "estimated_time": f"{estimated_seconds} seconds",
                            "resource_uri": f"resource://browser_task/{task_id}",
                            "sleep_command": "sleep 5",
                            "instruction": "Use the terminal command 'sleep 5' to wait 5 seconds between status checks. IMPORTANT: Always use exactly 5 seconds, no more and no less.",
                        },
                        indent=2,
                    ),
                )
            ]

        # Handle browser_get_result tool
        elif name == "browser_get_result":
            # Get result of async task
            if "task_id" not in arguments:
                raise ValueError("Missing required argument 'task_id'")

            task_id = arguments["task_id"]

            if task_id not in task_store:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {"error": "Task not found", "task_id": task_id}, indent=2
                        ),
                    )
                ]

            # Get the current task data
            task_data = task_store[task_id].copy()

            # If task is still running, add simple guidance
            if task_data["status"] == "running":
                # Add a simple next check suggestion
                progress = task_data.get("progress", {})
                current_step = progress.get("current_step", 0)

                if current_step > 0:
                    # Simple message based on current step
                    task_data["message"] = (
                        f"Task is running (step {current_step}). Wait 5 seconds before checking again."
                    )
                    task_data["sleep_command"] = "sleep 5"
                    task_data["instruction"] = (
                        "Use the terminal command 'sleep 5' to wait 5 seconds before checking again. IMPORTANT: Always use exactly 5 seconds, no more and no less."
                    )
                else:
                    task_data["message"] = (
                        "Task is starting. Wait 5 seconds before checking again."
                    )
                    task_data["sleep_command"] = "sleep 5"
                    task_data["instruction"] = (
                        "Use the terminal command 'sleep 5' to wait 5 seconds before checking again. IMPORTANT: Always use exactly 5 seconds, no more and no less."
                    )

            # Return current task status and result if available
            return [
                types.TextContent(type="text", text=json.dumps(task_data, indent=2))
            ]

        else:
            raise ValueError(f"Unknown tool: {name}")

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="browser_use",
                description="Performs a browser action and returns a task ID for async execution",
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
            ),
            types.Tool(
                name="browser_get_result",
                description="Gets the result of an asynchronous browser task",
                inputSchema={
                    "type": "object",
                    "required": ["task_id"],
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "ID of the task to get results for",
                        }
                    },
                },
            ),
        ]

    @app.list_resources()
    async def list_resources() -> list[types.Resource]:
        # List all completed tasks as resources
        resources = []
        for task_id, task_data in task_store.items():
            if task_data["status"] in ["completed", "failed"]:
                resources.append(
                    types.Resource(
                        uri=f"resource://browser_task/{task_id}",
                        title=f"Browser Task Result: {task_id[:8]}",
                        description=f"Result of browser task for URL: {task_data.get('url', 'unknown')}",
                    )
                )
        return resources

    @app.read_resource()
    async def read_resource(uri: str) -> list[types.ResourceContents]:
        # Extract task ID from URI
        if not uri.startswith("resource://browser_task/"):
            return [
                types.ResourceContents(
                    type="text",
                    text=json.dumps(
                        {"error": f"Invalid resource URI: {uri}"}, indent=2
                    ),
                )
            ]

        task_id = uri.replace("resource://browser_task/", "")
        if task_id not in task_store:
            return [
                types.ResourceContents(
                    type="text",
                    text=json.dumps({"error": f"Task not found: {task_id}"}, indent=2),
                )
            ]

        # Return task data
        return [
            types.ResourceContents(
                type="text", text=json.dumps(task_store[task_id], indent=2)
            )
        ]

    # Add cleanup_old_tasks function to app for later scheduling
    app.cleanup_old_tasks = cleanup_old_tasks

    return app


@click.command()
@click.option("--port", default=8000, help="Port to listen on for SSE")
@click.option(
    "--chrome-path",
    default=None,
    help="Path to Chrome executable",
)
@click.option(
    "--window-width",
    default=1280,
    help="Browser window width",
)
@click.option(
    "--window-height",
    default=1100,
    help="Browser window height",
)
@click.option(
    "--locale",
    default="en-US",
    help="Browser locale",
)
@click.option(
    "--task-expiry-minutes",
    default=60,
    help="Minutes after which tasks are considered expired",
)
def main(
    port: int,
    chrome_path: str,
    window_width: int,
    window_height: int,
    locale: str,
    task_expiry_minutes: int,
) -> int:
    """
    Run the browser-use MCP server.

    This function initializes the browser context, creates the MCP server,
    and runs it with the SSE transport.
    """
    global browser, context, config, browser_context_healthy

    # Use Chrome path from command line arg, environment variable, or None
    chrome_executable_path = chrome_path or os.environ.get("CHROME_PATH")
    if chrome_executable_path:
        logger.info(f"Using Chrome path: {chrome_executable_path}")
    else:
        logger.info(
            "No Chrome path specified, letting Playwright use its default browser"
        )

    # Initialize browser context
    try:
        # Browser context configuration
        config = BrowserContextConfig(
            wait_for_network_idle_page_load_time=0.6,
            maximum_wait_page_load_time=1.2,
            minimum_wait_page_load_time=0.2,
            browser_window_size={"width": window_width, "height": window_height},
            locale=locale,
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36",
            highlight_elements=True,
            viewport_expansion=0,
        )

        # Initialize browser and context directly
        browser_config = BrowserConfig(
            extra_chromium_args=[
                "--no-sandbox",
                "--disable-gpu",
                "--disable-software-rasterizer",
                "--disable-dev-shm-usage",
                "--remote-debugging-port=9222",
            ],
        )

        # Only set chrome_instance_path if we actually found a path
        if chrome_executable_path:
            browser_config.chrome_instance_path = chrome_executable_path

        browser = Browser(config=browser_config)
        context = BrowserContext(browser=browser, config=config)
        browser_context_healthy = True
        logger.info("Browser context initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize browser context: {str(e)}")
        return 1

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

    # Create MCP server
    app = create_mcp_server(
        llm=llm,
        task_expiry_minutes=task_expiry_minutes,
    )

    # Set up SSE transport
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.routing import Mount, Route
    import uvicorn

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

    # Add a startup event to initialize the browser
    @starlette_app.on_event("startup")
    async def startup_event():
        logger.info("Starting browser context...")
        await reset_browser_context()
        logger.info("Browser context started")

        # Start background task cleanup
        asyncio.create_task(app.cleanup_old_tasks())
        logger.info("Task cleanup process scheduled")

    # Add a shutdown event to clean up browser resources
    @starlette_app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Shutting down browser context...")
        try:
            await browser.close()
            logger.info("Browser context closed successfully")
        except Exception as e:
            logger.error(f"Error closing browser: {str(e)}")

    # Run uvicorn server
    uvicorn.run(starlette_app, host="0.0.0.0", port=port)

    return 0


if __name__ == "__main__":
    main()
