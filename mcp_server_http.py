from datetime import date

from fastmcp import FastMCP


mcp = FastMCP("MCPHttpServer")


@mcp.tool
def greet(name: str) -> str:
    """This is great tool, and based on the name, it is going to greet thee user."""
    return f"Hello {name}, Welome to my home ! How are you doing today ?"

@mcp.tool
def addition(num1: int, num2: int) -> int:
    """Add two integers and return the result."""
    return num1 + num2




if __name__ == "__main__":
    mcp.run(transport='http', host='127.0.0.1', port=8000, show_banner=False)