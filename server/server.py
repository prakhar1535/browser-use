import os
import click
import asyncio
import anyio
from dotenv import load_dotenv
import logging
import os.path

# Import from the browser-use-mcp-server package
from browser_use_mcp_server.server import (
    initialize_browser_context,
    create_mcp_server,
)
from langchain_openai import ChatOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


@click.command()
@click.option("--port", default=8000, help="Port to listen on for SSE")
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    help="Transport type",
)
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
    transport: str,
    chrome_path: str,
    window_width: int,
    window_height: int,
    locale: str,
    task_expiry_minutes: int,
) -> int:
    """Run the browser-use MCP server."""
    # If Chrome path is explicitly provided, use it
    if chrome_path and os.path.exists(chrome_path):
        chrome_executable_path = chrome_path
        logger.info(f"Using explicitly provided Chrome path: {chrome_executable_path}")
    else:
        # Try to find Playwright's installed Chromium first
        import os.path
        import platform
        from pathlib import Path
        import subprocess
        
        # Try to get Playwright browsers directory
        home_dir = str(Path.home())
        system = platform.system()
        
        if system == "Darwin":  # macOS
            playwright_browsers_path = os.path.join(home_dir, "Library", "Caches", "ms-playwright")
        elif system == "Linux":
            playwright_browsers_path = os.path.join(home_dir, ".cache", "ms-playwright")
        elif system == "Windows":
            playwright_browsers_path = os.path.join(home_dir, "AppData", "Local", "ms-playwright")
        else:
            playwright_browsers_path = None
        
        # Try to find the Chromium executable in the Playwright directory
        chromium_executable_path = None
        if playwright_browsers_path and os.path.exists(playwright_browsers_path):
            logger.info(f"Found Playwright browsers directory at {playwright_browsers_path}")
            # Look for chromium directories
            try:
                for root, dirs, files in os.walk(playwright_browsers_path):
                    for dir in dirs:
                        if "chromium" in dir.lower():
                            # Check for executable in this directory
                            if system == "Darwin":  # macOS
                                exec_path = os.path.join(root, dir, "chrome-mac", "Chromium.app", "Contents", "MacOS", "Chromium")
                            elif system == "Linux":
                                exec_path = os.path.join(root, dir, "chrome-linux", "chrome")
                            elif system == "Windows":
                                exec_path = os.path.join(root, dir, "chrome-win", "chrome.exe")
                            else:
                                continue
                                
                            if os.path.exists(exec_path):
                                chromium_executable_path = exec_path
                                logger.info(f"Found Playwright Chromium at {chromium_executable_path}")
                                break
                    if chromium_executable_path:
                        break
            except Exception as e:
                logger.warning(f"Error searching for Playwright Chromium: {str(e)}")
        
        # If Playwright Chromium not found, try standard locations
        if not chromium_executable_path:
            # Try to find Chromium/Chrome in common locations
            potential_paths = [
                # Environment variable
                os.environ.get("CHROME_PATH"),
                
                # Common macOS paths
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                "/Applications/Chromium.app/Contents/MacOS/Chromium",
                
                # Common Linux paths
                "/usr/bin/chromium",
                "/usr/bin/chromium-browser",
                "/usr/bin/google-chrome",
                "/usr/bin/google-chrome-stable",
            ]
            
            for path in potential_paths:
                if path and os.path.exists(path):
                    logger.info(f"Found browser at {path}")
                    chromium_executable_path = path
                    break
        
        # Use the found path or try with Playwright's default
        if chromium_executable_path:
            chrome_executable_path = chromium_executable_path
            logger.info(f"Using browser executable at: {chrome_executable_path}")
        else:
            # If no specific path found, let Playwright find its own browser
            logger.warning("No Chrome/Chromium path found, will let Playwright use its default browser")
            chrome_executable_path = None
    
    # Initialize browser context
    try:
        # Using the approach from backup/server.py
        from browser_use.browser.context import BrowserContextConfig, BrowserContext
        from browser_use.browser.browser import Browser, BrowserConfig
        
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
        logger.info("Browser context initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize browser context: {str(e)}")
        return 1
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
    
    # Create MCP server
    app = create_mcp_server(
        context=context,
        llm=llm,
        task_expiry_minutes=task_expiry_minutes,
    )
    
    if transport == "sse":
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
                raise
                
        starlette_app = Starlette(
            debug=True,
            routes=[
                Route("/sse", endpoint=handle_sse),
                Mount("/messages/", app=sse.handle_post_message),
            ],
        )
        
        # Add a startup event to initialize the browser and start task cleanup
        @starlette_app.on_event("startup")
        async def startup_event():
            logger.info("Starting server and scheduling cleanup...")
            
            # Start the cleanup task now that we have an event loop
            if hasattr(app, 'cleanup_old_tasks'):
                asyncio.create_task(app.cleanup_old_tasks())
                logger.info("Task cleanup process scheduled")
        
        # Add a shutdown event to clean up browser resources
        @starlette_app.on_event("shutdown")
        async def shutdown_event():
            logger.info("Shutting down server and cleaning up resources...")
            try:
                await context.browser.close()
                logger.info("Browser context closed successfully")
            except Exception as e:
                logger.error(f"Error closing browser: {str(e)}")
        
        uvicorn.run(starlette_app, host="0.0.0.0", port=port)
    else:
        from mcp.server.stdio import stdio_server
        
        async def arun():
            try:
                # Start the cleanup task now that we have an event loop
                if hasattr(app, 'cleanup_old_tasks'):
                    asyncio.create_task(app.cleanup_old_tasks())
                    logger.info("Task cleanup process scheduled")
                
                async with stdio_server() as streams:
                    await app.run(
                        streams[0], streams[1], app.create_initialization_options()
                    )
            except Exception as e:
                logger.error(f"Error in arun: {str(e)}")
            finally:
                # Clean up resources
                try:
                    await context.browser.close()
                except Exception as e:
                    logger.error(f"Error cleaning up resources: {str(e)}")
                    
        anyio.run(arun)
        
    return 0


if __name__ == "__main__":
    main()
