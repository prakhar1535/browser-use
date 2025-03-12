"""
Command-line interface for browser-use-mcp-server.
"""

import os
import click
import asyncio
import logging
import uvicorn
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import JSONResponse, StreamingResponse
from typing import Dict, Any, Optional
import sys

from langchain_openai import ChatOpenAI
from mcp.server.lowlevel import Server

from .server import (
    initialize_browser_context,
    create_mcp_server,
    check_browser_health,
    reset_browser_context,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AsyncStdinReader:
    """Async wrapper for stdin."""

    async def receive(self) -> bytes:
        line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
        return line.encode()


class AsyncStdoutWriter:
    """Async wrapper for stdout."""

    async def send(self, data: bytes) -> None:
        text = data.decode()
        sys.stdout.write(text)
        sys.stdout.flush()
        await asyncio.sleep(0)  # Yield control back to the event loop


@click.group()
def cli():
    """Browser MCP Server CLI."""
    pass


@cli.command()
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
    help="Path to Chrome executable (defaults to CHROME_PATH env var)",
)
@click.option(
    "--model",
    default="gpt-4o",
    help="OpenAI model to use",
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
    "--task-expiry-minutes",
    default=60,
    help="Minutes after which tasks expire",
)
def start(
    port: int,
    transport: str,
    chrome_path: Optional[str],
    model: str,
    window_width: int,
    window_height: int,
    task_expiry_minutes: int,
) -> int:
    """Start the browser MCP server."""
    # Record tasks for SSE transport
    task_store: Dict[str, Any] = {}

    # Set up browser context and LLM
    if chrome_path is None:
        chrome_path = os.environ.get("CHROME_PATH")

    try:
        logger.info(
            f"Initializing browser context with Chrome path: {chrome_path or 'default'}"
        )
        context = initialize_browser_context(
            chrome_path=chrome_path,
            window_width=window_width,
            window_height=window_height,
        )
        logger.info("Browser context initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize browser context: {e}")
        return 1

    try:
        logger.info(f"Initializing LLM with model: {model}")
        llm = ChatOpenAI(model=model, temperature=0.0)
        logger.info("LLM initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        return 1

    try:
        # Create MCP server
        logger.info("Creating MCP server")
        app = create_mcp_server(
            context=context,
            llm=llm,
            custom_task_store=task_store,
            task_expiry_minutes=task_expiry_minutes,
        )
        logger.info("MCP server created successfully")
    except Exception as e:
        logger.error(f"Failed to create MCP server: {e}")
        return 1

    if transport == "stdio":
        # Run the server with stdio transport
        logger.info("Starting browser MCP server with stdio transport")
        return asyncio.run(_run_stdio(app))

    else:
        # Set up Starlette app for SSE transport
        async def handle_sse(request):
            """Handle SSE connections."""
            logger.info(f"New SSE connection from {request.client}")
            logger.info(f"Request headers: {request.headers}")

            # Create a queue for sending messages
            send_queue = asyncio.Queue()

            # Define message handlers for MCP server
            class SSEReadStream:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    pass

                async def receive(self) -> bytes:
                    # For SSE, we don't receive anything from client
                    # Just block indefinitely
                    future = asyncio.Future()
                    await future  # This will block forever
                    return b""  # Never reached

            class SSEWriteStream:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    pass

                async def send(self, data: bytes) -> None:
                    # Queue the message to be sent over SSE
                    await send_queue.put(data)

            # Create async generator to stream SSE responses
            async def stream_response():
                """Stream SSE responses."""
                logger.info("Setting up SSE stream")

                # Start MCP server in background
                read_stream = SSEReadStream()
                write_stream = SSEWriteStream()

                server_task = asyncio.create_task(
                    app.run(
                        read_stream=read_stream,
                        write_stream=write_stream,
                        initialization_options={},
                    )
                )

                try:
                    # Send initial connected event
                    logger.info("Sending initial connected event")
                    yield b"event: connected\ndata: {}\n\n"

                    # Stream messages from the queue
                    logger.info("Starting to stream messages")
                    while True:
                        message = await send_queue.get()
                        logger.info(f"Sending message: {message[:100]}...")
                        data = f"data: {message.decode()}\n\n"
                        yield data.encode("utf-8")
                        send_queue.task_done()
                except Exception as e:
                    logger.error(f"SSE streaming error: {e}")
                finally:
                    # Clean up
                    server_task.cancel()
                    logger.info("SSE connection closed")

            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        async def health_check(request):
            """Health check endpoint."""
            try:
                # Check browser health
                healthy = await check_browser_health(context)
                return JSONResponse({"status": "healthy" if healthy else "unhealthy"})
            except Exception as e:
                return JSONResponse(
                    {"status": "error", "message": str(e)}, status_code=500
                )

        async def reset_context(request):
            """Reset browser context endpoint."""
            try:
                # Reset browser context
                await reset_browser_context(context)
                return JSONResponse(
                    {"status": "success", "message": "Browser context reset"}
                )
            except Exception as e:
                return JSONResponse(
                    {"status": "error", "message": str(e)}, status_code=500
                )

        # Define startup and shutdown events
        async def startup_event():
            """Run on server startup."""
            logger.info("Starting server...")

            # Start task cleanup job
            asyncio.create_task(cleanup_old_tasks())

            logger.info(f"Server started on port {port}")

        async def shutdown_event():
            """Run on server shutdown."""
            logger.info("Shutting down server...")

            try:
                # Close the browser
                await context.browser.close()
                logger.info("Browser closed successfully")
            except Exception as e:
                logger.error(f"Error closing browser: {e}")

            logger.info("Server shut down")

        async def cleanup_old_tasks():
            """Periodically clean up expired tasks."""
            from datetime import datetime

            while True:
                try:
                    # Check for expired tasks every minute
                    await asyncio.sleep(60)

                    # Get current time
                    now = datetime.now()

                    # Check each task
                    expired_tasks = []
                    for task_id, task in task_store.items():
                        if "expiry_time" in task:
                            # Parse expiry time
                            expiry_time = datetime.fromisoformat(task["expiry_time"])

                            # Check if expired
                            if now > expiry_time:
                                expired_tasks.append(task_id)

                    # Remove expired tasks
                    for task_id in expired_tasks:
                        logger.info(f"Removing expired task {task_id}")
                        task_store.pop(task_id, None)

                except Exception as e:
                    logger.error(f"Error cleaning up old tasks: {e}")

        # Create Starlette app with routes
        routes = [
            Route("/sse", endpoint=handle_sse, methods=["GET"]),
            Route("/health", endpoint=health_check, methods=["GET"]),
            Route("/reset", endpoint=reset_context, methods=["POST"]),
        ]

        starlette_app = Starlette(
            routes=routes,
            on_startup=[startup_event],
            on_shutdown=[shutdown_event],
            debug=True,
        )

        # Run with uvicorn
        logger.info(f"Starting browser MCP server with SSE transport on port {port}")
        uvicorn.run(starlette_app, host="0.0.0.0", port=port)

        return 0


async def _run_stdio(app: Server) -> int:
    """Run the server using stdio transport."""
    try:
        stdin_reader = AsyncStdinReader()
        stdout_writer = AsyncStdoutWriter()

        # Create initialization options
        initialization_options = {}

        # Run the server
        await app.run(
            read_stream=stdin_reader,
            write_stream=stdout_writer,
            initialization_options=initialization_options,
        )
        return 0
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        return 0
    except Exception as e:
        logger.error(f"Error running server: {e}")
        return 1


if __name__ == "__main__":
    cli()
