"""
Browser Use MCP Server

This module implements an MCP (Model-Control-Protocol) server for browser automation
using the browser_use library. It provides functionality to interact with a browser instance
via an async task queue, allowing for long-running browser tasks to be executed asynchronously
while providing status updates and results.

The server supports Server-Sent Events (SSE) for web-based interfaces.
"""

# Standard library imports
import os
import asyncio
import json
import logging
import traceback
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
import requests
# Third-party imports
import click
from dotenv import load_dotenv

# Browser-use library imports
from browser_use import Agent
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig

# MCP server components
from mcp.server import Server
import mcp.types as types

# LLM provider
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseLanguageModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def init_configuration() -> Dict[str, any]:
    """
    Initialize configuration from environment variables with defaults.

    Returns:
        Dictionary containing all configuration parameters
    """
    config = {
        # Browser window settings
        "DEFAULT_WINDOW_WIDTH": int(os.environ.get("BROWSER_WINDOW_WIDTH", 1280)),
        "DEFAULT_WINDOW_HEIGHT": int(os.environ.get("BROWSER_WINDOW_HEIGHT", 1100)),
        # Browser config settings
        "DEFAULT_LOCALE": os.environ.get("BROWSER_LOCALE", "en-US"),
        "DEFAULT_USER_AGENT": os.environ.get(
            "BROWSER_USER_AGENT",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36",
        ),
        # Task settings
        "DEFAULT_TASK_EXPIRY_MINUTES": int(os.environ.get("TASK_EXPIRY_MINUTES", 60)),
        "DEFAULT_ESTIMATED_TASK_SECONDS": int(
            os.environ.get("ESTIMATED_TASK_SECONDS", 60)
        ),
        "CLEANUP_INTERVAL_SECONDS": int(
            os.environ.get("CLEANUP_INTERVAL_SECONDS", 3600)
        ),  # 1 hour
        "MAX_AGENT_STEPS": int(os.environ.get("MAX_AGENT_STEPS", 10)),
        # Browser arguments
        "BROWSER_ARGS": [
            "--no-sandbox",
            "--disable-gpu",
            "--disable-software-rasterizer",
            "--disable-dev-shm-usage",
            "--remote-debugging-port=0",  # Use random port to avoid conflicts
        ],
    }

    return config


# Initialize configuration
CONFIG = init_configuration()

# Task storage for async operations
task_store: Dict[str, Dict[str, Any]] = {}


async def create_browser_context_for_task(
    chrome_path: Optional[str] = None,
    window_width: int = CONFIG["DEFAULT_WINDOW_WIDTH"],
    window_height: int = CONFIG["DEFAULT_WINDOW_HEIGHT"],
    locale: str = CONFIG["DEFAULT_LOCALE"],
) -> Tuple[Browser, BrowserContext]:
    """
    Create a fresh browser and context for a task.

    This function creates an isolated browser instance and context
    with proper configuration for a single task using Anchor Browser.

    Args:
        chrome_path: Path to Chrome executable (not used with Anchor Browser)
        window_width: Browser window width
        window_height: Browser window height
        locale: Browser locale

    Returns:
        A tuple containing the browser instance and browser context

    Raises:
        Exception: If browser or context creation fails
    """
    try:
        # Get Anchor API key from environment variables
        anchor_api_key = os.environ.get("ANCHOR_API_KEY")
        if not anchor_api_key:
            raise ValueError("ANCHOR_API_KEY not found in environment variables")
            
        # Create an Anchor browser session
        response = requests.post(
            "https://api.anchorbrowser.io/api/sessions",
            headers={
                "anchor-api-key": anchor_api_key,
                "Content-Type": "application/json",
            },
            json={
                "headless": False,  # Use headless false to view the browser
            }
        )
        
        # Check if request was successful
        response.raise_for_status()
        session_data = response.json()
        
        # Log the session creation
        logger.info(f"Created Anchor browser session: {session_data['id']}")
        logger.info(f"Live view URL: {session_data.get('live_view_url', 'N/A')}")
        
        # Store the session data for later reference
        session_id = session_data['id']
        
        # Create browser configuration with CDP URL to connect to Anchor browser
        browser_config = BrowserConfig(
            cdp_url=f"wss://connect.anchorbrowser.io?apiKey={anchor_api_key}&sessionId={session_id}"
        )

        # Create browser instance
        browser = Browser(config=browser_config)

        # Create context configuration
        context_config = BrowserContextConfig(
            wait_for_network_idle_page_load_time=0.6,
            maximum_wait_page_load_time=1.2,
            minimum_wait_page_load_time=0.2,
            browser_window_size={"width": window_width, "height": window_height},
            locale=locale,
            user_agent=CONFIG["DEFAULT_USER_AGENT"],
            highlight_elements=True,
            viewport_expansion=0,
        )

        # Create context with the browser
        context = BrowserContext(browser=browser, config=context_config)
        
        # Store the Anchor browser session information in the browser object for later access
        browser.anchor_session = {
            "session_id": session_id,
            "page_id": session_data.get('page_id'),
            "live_view_url": session_data.get('live_view_url')
        }

        return browser, context
    except Exception as e:
        logger.error(f"Error creating Anchor browser context: {str(e)}")
        raise


async def run_browser_task_async(
    task_id: str,
    url: str,
    action: str,
    llm: BaseLanguageModel,
    window_width: int = CONFIG["DEFAULT_WINDOW_WIDTH"],
    window_height: int = CONFIG["DEFAULT_WINDOW_HEIGHT"],
    locale: str = CONFIG["DEFAULT_LOCALE"],
) -> None:
    """
    Run a browser task asynchronously and store the result.

    This function executes a browser automation task with the given URL and action,
    and updates the task store with progress and results.

    Args:
        task_id: Unique identifier for the task
        url: URL to navigate to
        action: Action to perform after navigation
        llm: Language model to use for browser agent
        window_width: Browser window width
        window_height: Browser window height
        locale: Browser locale
    """
    browser = None
    context = None

    try:
        # Update task status to running
        task_store[task_id]["status"] = "running"
        task_store[task_id]["start_time"] = datetime.now().isoformat()
        task_store[task_id]["progress"] = {
            "current_step": 0,
            "total_steps": 0,
            "steps": [],
        }

        # Define step callback function with the correct signature
        async def step_callback(
            browser_state: Any, agent_output: Any, step_number: int
        ) -> None:
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
        async def done_callback(history: Any) -> None:
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

        # Check if we already have an Anchor browser session in the task store
        existing_anchor_session = task_store[task_id].get("anchor_session")
        
        if existing_anchor_session and existing_anchor_session.get("session_id"):
            # Use the existing Anchor browser session
            session_id = existing_anchor_session["session_id"]
            logger.info(f"Using existing Anchor browser session: {session_id}")
            
            # Get Anchor API key from environment variables
            anchor_api_key = os.environ.get("ANCHOR_API_KEY")
            if not anchor_api_key:
                raise ValueError("ANCHOR_API_KEY not found in environment variables")
                
            # Create browser configuration with CDP URL to connect to existing Anchor browser
            browser_config = BrowserConfig(
                cdp_url=f"wss://connect.anchorbrowser.io?apiKey={anchor_api_key}&sessionId={session_id}"
            )

            # Create browser instance
            browser = Browser(config=browser_config)

            # Create context configuration
            context_config = BrowserContextConfig(
                wait_for_network_idle_page_load_time=0.6,
                maximum_wait_page_load_time=1.2,
                minimum_wait_page_load_time=0.2,
                browser_window_size={"width": window_width, "height": window_height},
                locale=locale,
                user_agent=CONFIG["DEFAULT_USER_AGENT"],
                highlight_elements=True,
                viewport_expansion=0,
            )

            # Create context with the browser
            context = BrowserContext(browser=browser, config=context_config)
            
            # Store the Anchor browser session information in the browser object for later access
            browser.anchor_session = existing_anchor_session
        else:
            # Create a fresh browser and context for this task
            browser, context = await create_browser_context_for_task(
                window_width=window_width,
                window_height=window_height,
                locale=locale,
            )
            
            # Store Anchor browser session info in the task store
            if hasattr(browser, 'anchor_session'):
                task_store[task_id]["anchor_session"] = browser.anchor_session

        # Create agent with the fresh context
        agent = Agent(
            task=f"First, navigate to {url}. Then, {action}",
            llm=llm,
            browser_context=context,
            register_new_step_callback=step_callback,
            register_done_callback=done_callback,
        )

        # Run the agent with a reasonable step limit
        agent_result = await agent.run(max_steps=CONFIG["MAX_AGENT_STEPS"])

        # Get the final result
        final_result = agent_result.final_result()

        # Check if we have a valid result
        if final_result and hasattr(final_result, "raise_for_status"):
            final_result.raise_for_status()
            result_text = str(final_result.text)
        else:
            result_text = (
                str(final_result) if final_result else "No final result available"
            )

        # Gather essential information from the agent history
        is_successful = agent_result.is_successful()
        has_errors = agent_result.has_errors()
        errors = agent_result.errors()
        urls_visited = agent_result.urls()
        action_names = agent_result.action_names()
        extracted_content = agent_result.extracted_content()
        steps_taken = agent_result.number_of_steps()

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

        # Store the error
        task_store[task_id]["status"] = "failed"
        task_store[task_id]["end_time"] = datetime.now().isoformat()
        task_store[task_id]["error"] = str(e)
        task_store[task_id]["traceback"] = tb

    finally:
        # Clean up browser resources
        try:
            if context:
                await context.close()
            if browser:
                # If using Anchor Browser, we may want to close the session via API
                if hasattr(browser, 'anchor_session'):
                    session_id = browser.anchor_session.get('session_id')
                    if session_id:
                        # Consider adding code to close Anchor session via API if needed
                        logger.info(f"Note: Anchor browser session {session_id} may need explicit cleanup via API")
                await browser.close()
            logger.info(f"Browser resources for task {task_id} cleaned up")
        except Exception as e:
            logger.error(
                f"Error cleaning up browser resources for task {task_id}: {str(e)}"
            )


async def cleanup_old_tasks() -> None:
    """
    Periodically clean up old completed tasks to prevent memory leaks.

    This function runs continuously in the background, removing tasks that have been
    completed or failed for more than 1 hour to conserve memory.
    """
    while True:
        try:
            # Sleep first to avoid cleaning up tasks too early
            await asyncio.sleep(CONFIG["CLEANUP_INTERVAL_SECONDS"])

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


def create_mcp_server(
    llm: BaseLanguageModel,
    task_expiry_minutes: int = CONFIG["DEFAULT_TASK_EXPIRY_MINUTES"],
    window_width: int = CONFIG["DEFAULT_WINDOW_WIDTH"],
    window_height: int = CONFIG["DEFAULT_WINDOW_HEIGHT"],
    locale: str = CONFIG["DEFAULT_LOCALE"],
) -> Server:
    """
    Create and configure an MCP server for browser interaction.

    Args:
        llm: The language model to use for browser agent
        task_expiry_minutes: Minutes after which tasks are considered expired
        window_width: Browser window width
        window_height: Browser window height
        locale: Browser locale

    Returns:
        Configured MCP server instance
    """
    # Create MCP server instance
    app = Server("browser_use")

    @app.call_tool()
    async def call_tool(
        name: str, arguments: dict
    ) -> list[Union[types.TextContent, types.ImageContent, types.EmbeddedResource]]:
        """
        Handle tool calls from the MCP client.

        Args:
            name: The name of the tool to call
            arguments: The arguments to pass to the tool

        Returns:
            A list of content objects to return to the client

        Raises:
            ValueError: If required arguments are missing
        """
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
                    task_id=task_id,
                    url=arguments["url"],
                    action=arguments["action"],
                    llm=llm,
                    window_width=window_width,
                    window_height=window_height,
                    locale=locale,
                )
            )

            # Return task ID immediately with explicit sleep instruction
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "task_id": task_id,
                            "status": "pending",
                            "message": f"Browser task started. Please wait for {CONFIG['DEFAULT_ESTIMATED_TASK_SECONDS']} seconds, then check the result using browser_get_result or the resource URI. Always wait exactly 5 seconds between status checks.",
                            "estimated_time": f"{CONFIG['DEFAULT_ESTIMATED_TASK_SECONDS']} seconds",
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
        """
        List the available tools for the MCP client.

        Returns:
            A list of tool definitions
        """
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
        """
        List the available resources for the MCP client.

        Returns:
            A list of resource definitions
        """
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
        """
        Read a resource for the MCP client.

        Args:
            uri: The URI of the resource to read

        Returns:
            The contents of the resource
        """
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


# Add the API routes function
def create_api_routes(app):
    """
    Create HTTP API routes for direct browser automation.
    
    Args:
        app: The MCP server instance
        
    Returns:
        List of Starlette Route objects
    """
    async def api_start_browser_task(request):
        """
        HTTP endpoint to start a browser task.
        
        Request body should contain:
        {
            "url": "https://example.com",
            "action": "Extract the main heading"
        }
        """
        try:
            # Parse JSON request body
            body = await request.json()
            
            # Validate required fields
            if "url" not in body:
                return JSONResponse({"error": "Missing required field 'url'"}, status_code=400)
            if "action" not in body:
                return JSONResponse({"error": "Missing required field 'action'"}, status_code=400)
            
            # Generate task ID
            task_id = str(uuid.uuid4())
            
            # Get custom parameters if provided
            window_width = body.get("window_width", CONFIG["DEFAULT_WINDOW_WIDTH"])
            window_height = body.get("window_height", CONFIG["DEFAULT_WINDOW_HEIGHT"])
            locale = body.get("locale", CONFIG["DEFAULT_LOCALE"])
            
            # Create LLM instance (or reuse from app)
            llm = ChatOpenAI(model=body.get("model", "gpt-4o"), temperature=body.get("temperature", 0.0))
            
            # ENHANCEMENT: Create the Anchor browser session synchronously before returning
            anchor_session_info = None
            try:
                # Get Anchor API key from environment variables
                anchor_api_key = os.environ.get("ANCHOR_API_KEY")
                if not anchor_api_key:
                    raise ValueError("ANCHOR_API_KEY not found in environment variables")
                    
                # Create an Anchor browser session
                response = requests.post(
                    "https://api.anchorbrowser.io/api/sessions",
                    headers={
                        "anchor-api-key": anchor_api_key,
                        "Content-Type": "application/json",
                    },
                    json={
                        "headless": False,  # Use headless false to view the browser
                    }
                )
                
                # Check if request was successful
                response.raise_for_status()
                session_data = response.json()
                
                # Log the session creation
                logger.info(f"Created Anchor browser session: {session_data['id']}")
                logger.info(f"Live view URL: {session_data.get('live_view_url', 'N/A')}")
                
                # Store the session data for later reference
                anchor_session_info = {
                    "session_id": session_data['id'],
                    "page_id": session_data.get('page_id'),
                    "live_view_url": session_data.get('live_view_url')
                }
            except Exception as e:
                logger.error(f"Error creating Anchor browser session: {str(e)}")
                # Continue without Anchor session info - task will try again during execution
            
            # Initialize task in store with Anchor session info if available
            task_info = {
                "id": task_id,
                "status": "pending",
                "url": body["url"],
                "action": body["action"],
                "created_at": datetime.now().isoformat()
            }
            
            # Add anchor session info if we have it
            if anchor_session_info:
                task_info["anchor_session"] = anchor_session_info
            
            # Store task info
            task_store[task_id] = task_info
            
            # Start task in background
            asyncio.create_task(
                run_browser_task_async(
                    task_id=task_id,
                    url=body["url"],
                    action=body["action"],
                    llm=llm,
                    window_width=window_width,
                    window_height=window_height,
                    locale=locale,
                )
            )
            
            # Set wait parameter
            wait = body.get("wait", False)
            
            if wait:
                # If wait=True, we'll wait for the task to complete
                # This makes the API call synchronous
                max_wait_seconds = body.get("max_wait_seconds", 120)
                start_time = datetime.now()
                
                # Wait for task to complete or timeout
                while (datetime.now() - start_time).total_seconds() < max_wait_seconds:
                    # Check if task completed
                    if task_id in task_store and task_store[task_id]["status"] in ["completed", "failed"]:
                        # Return full result
                        return JSONResponse(task_store[task_id])
                    
                    # Sleep briefly before checking again
                    await asyncio.sleep(0.5)
                
                # If we get here, task timed out
                return JSONResponse({
                    "task_id": task_id,
                    "status": "timeout",
                    "message": f"Task is still running after {max_wait_seconds} seconds. Use GET /api/browser/tasks/{task_id} to check status later.",
                    "status_url": f"/api/browser/tasks/{task_id}",
                    "anchor_session": task_store[task_id].get("anchor_session")
                })
            else:
                # Return task ID and Anchor session info for async workflow
                response_data = {
                    "task_id": task_id,
                    "status": "pending",
                    "message": "Browser task started. Check status at GET /api/browser/tasks/{task_id}",
                    "status_url": f"/api/browser/tasks/{task_id}"
                }
                
                # Add anchor session info to response if available
                if "anchor_session" in task_store[task_id]:
                    response_data["anchor_session"] = task_store[task_id]["anchor_session"]
                    response_data["live_view_url"] = task_store[task_id]["anchor_session"].get("live_view_url")
                
                return JSONResponse(response_data)
            
        except Exception as e:
            logger.error(f"Error in API start task: {str(e)}")
            return JSONResponse({"error": str(e)}, status_code=500)
    
    async def api_get_task_status(request):
        """
        HTTP endpoint to get the status of a browser task.
        """
        try:
            # Get task ID from path parameters
            task_id = request.path_params["task_id"]
            
            if task_id not in task_store:
                return JSONResponse({"error": "Task not found"}, status_code=404)
            
            # Return current task data
            return JSONResponse(task_store[task_id])
            
        except Exception as e:
            logger.error(f"Error in API get task status: {str(e)}")
            return JSONResponse({"error": str(e)}, status_code=500)
    
    async def api_list_tasks(request):
        """
        HTTP endpoint to list all browser tasks.
        """
        try:
            # Get query parameters
            status = request.query_params.get("status")
            limit = int(request.query_params.get("limit", 20))
            
            # Filter tasks by status if provided
            if status:
                filtered_tasks = {
                    task_id: task_data 
                    for task_id, task_data in task_store.items() 
                    if task_data["status"] == status
                }
            else:
                filtered_tasks = task_store
            
            # Sort by creation time (newest first)
            sorted_tasks = sorted(
                filtered_tasks.items(),
                key=lambda x: x[1].get("created_at", ""),
                reverse=True
            )
            
            # Apply limit
            limited_tasks = sorted_tasks[:limit]
            
            # Convert to list of task data
            task_list = [task_data for _, task_data in limited_tasks]
            
            return JSONResponse({
                "tasks": task_list,
                "total": len(filtered_tasks),
                "limit": limit
            })
            
        except Exception as e:
            logger.error(f"Error in API list tasks: {str(e)}")
            return JSONResponse({"error": str(e)}, status_code=500)
    
    # Define the API routes
    return [
        Route("/api/browser/tasks", endpoint=api_start_browser_task, methods=["POST"]),
        Route("/api/browser/tasks/{task_id}", endpoint=api_get_task_status, methods=["GET"]),
        Route("/api/browser/tasks", endpoint=api_list_tasks, methods=["GET"]),
    ]


# UPDATED MAIN FUNCTION WITH DEFAULT VALUES
def main(
    port: int = 8001,
    chrome_path: str = None,
    window_width: int = CONFIG["DEFAULT_WINDOW_WIDTH"],
    window_height: int = CONFIG["DEFAULT_WINDOW_HEIGHT"],
    locale: str = CONFIG["DEFAULT_LOCALE"],
    task_expiry_minutes: int = CONFIG["DEFAULT_TASK_EXPIRY_MINUTES"],
) -> int:
    """
    Run the browser-use MCP server.

    This function initializes the MCP server and runs it with the SSE transport.
    Each browser task will create its own isolated browser context.

    Args:
        port: Port to listen on for SSE
        chrome_path: Path to Chrome executable
        window_width: Browser window width
        window_height: Browser window height
        locale: Browser locale
        task_expiry_minutes: Minutes after which tasks are considered expired

    Returns:
        Exit code (0 for success)
    """
    # Store Chrome path in environment variable if provided
    if chrome_path:
        os.environ["CHROME_PATH"] = chrome_path
        logger.info(f"Using Chrome path: {chrome_path}")
    else:
        logger.info(
            "No Chrome path specified, letting Playwright use its default browser"
        )

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    # Create MCP server
    app = create_mcp_server(
        llm=llm,
        task_expiry_minutes=task_expiry_minutes,
        window_width=window_width,
        window_height=window_height,
        locale=locale,
    )

    # Set up SSE transport
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    import uvicorn

    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        """Handle SSE connections from clients."""
        try:
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await app.run(
                    streams[0], streams[1], app.create_initialization_options()
                )
        except Exception as e:
            logger.error(f"Error in handle_sse: {str(e)}")
            raise

    # Create API routes
    api_routes = create_api_routes(app)

    # Add CORS middleware
    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],  # For production, specify the allowed origins
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
        )
    ]

    # Create Starlette app with both SSE and API routes
    starlette_app = Starlette(
        debug=True,
        middleware=middleware,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
            # Add the API routes
            *api_routes,
        ],
    )

    # Add a startup event
    @starlette_app.on_event("startup")
    async def startup_event():
        """Initialize the server on startup."""
        logger.info("Starting MCP server with HTTP API...")

        # Sanity checks for critical configuration
        if port <= 0 or port > 65535:
            logger.error(f"Invalid port number: {port}")
            raise ValueError(f"Invalid port number: {port}")

        if window_width <= 0 or window_height <= 0:
            logger.error(f"Invalid window dimensions: {window_width}x{window_height}")
            raise ValueError(
                f"Invalid window dimensions: {window_width}x{window_height}"
            )

        if task_expiry_minutes <= 0:
            logger.error(f"Invalid task expiry minutes: {task_expiry_minutes}")
            raise ValueError(f"Invalid task expiry minutes: {task_expiry_minutes}")

        # Start background task cleanup
        asyncio.create_task(app.cleanup_old_tasks())
        logger.info("Task cleanup process scheduled")

    # Run uvicorn server
    uvicorn.run(starlette_app, host="0.0.0.0", port=port)