# Tool Library

## Overview

Tool Library is a dynamic Python library for managing and executing a variety of tools or functions. It facilitates the registration, execution, and tracking of tools, providing a flexible and extensible framework suitable for a wide range of applications. The library includes a FastAPI-based web service for interacting with the tool library via RESTful API.

## To Do List

 - Run Ray Functions.
 - Integrate with GPT (partial through the custom_gpts scripts, needs a permanent IP)
 - Add meaningful tools

## Installation

To use the Tool Library, ensure Python is installed on your system. Clone the repository and install the required dependencies:

```bash
git clone <repository_url>
cd tool_library
pip install -e .
```

## Usage

### Tool Library Class

Registering a Tool

```python

from tool_library.library import ToolLibrary
from tool_library.tools import *
from tool_library.factory import *

tool_library = ToolLibrary()
```

### Example Tools

```python

from pydantic import BaseModel

class ExampleModel(BaseModel):
    field1: int
    field2: str

def add_numbers(a: int, b: int) -> int:
    return a + b

def process_model(data: ExampleModel):
    return {"processed": True}

tool_library.register_tool("Add Numbers", "Adds two numbers", add_numbers)
tool_library.register_tool("ProcessModel", "Processes a Pydantic model", process_model)
```

### Executing a Tool

```python
tool_library.execute_tool("ProcessModel", {"field1": 10, "field2": "example"})
```

### Finding Tools

```python
tool_library.find_tools("Add")
```

## FastAPI Endpoints

### Registering a Tool


POST /register-tool/

Request Body:

```json
{
    "name": "example_tool",
    "description": "An example tool",
    "function": "example_function_name"
}
```

### Executing a Tool

POST /execute-tool/

Request Body:

```json
{
    "tool_name": "example_tool",
    "params": {
        "param1": "value1",
        "param2": "value2"
    }

}
```

### Getting Tool Statistics

GET /tool-stats/{tool_name}

### Finding Tools

POST /find-tools/

Request Body:

```json
{
    "query": "example"
}
```

### Getting All Tools

GET /get-tools/

### Removing a Tool

DELETE /remove-tool/{tool_name}

## Example Use Cases

    Data Processing: Manage different data processing algorithms.
    Automation Tasks: Automate repetitive tasks with encapsulated tools.
    Microservices: Integrate in a microservice architecture for diverse functionalities.

## Contributing

Contributions to the Tool Library project are welcome. Please follow the project's coding standards and submit your pull requests for review.