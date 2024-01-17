import json


def generate_custom_openapi_spec(
    url, original_spec, new_title, new_description, routes_to_keep=None
):
    """
    Generates a custom OpenAPI specification based on the provided original specification,
    title, description, and a list of routes to keep.

    :param original_spec: The original OpenAPI specification (as a dictionary).
    :param new_title: The new title for the OpenAPI specification.
    :param new_description: The new description for the OpenAPI specification.
    :param routes_to_keep: A list of routes (paths) to keep in the new specification.
    :return: A JSON string of the new OpenAPI specification.
    """
    new_spec = {
        "openapi": original_spec.get("openapi", "3.1.0"),
        "info": {
            "title": new_title,
            "version": original_spec.get("info", {}).get("version", "1.0.0"),
            "description": new_description,
        },
        "paths": {
            path: info
            for path, info in original_spec.get("paths", {}).items()
            if routes_to_keep is None or path in routes_to_keep
        },
        "components": original_spec.get("components", {}),
        "servers": [{"url": url}],
    }

    return json.dumps(new_spec, indent=2)


import requests


def fetch_openapi_spec(url):
    """
    Fetches the OpenAPI specification from a given URL.

    :param url: The URL from which to fetch the OpenAPI specification.
    :return: The OpenAPI specification as a dictionary if successful, None otherwise.
    """
    try:
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code
        except:
            response = requests.get(url + "/openapi.json")
            response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code

        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching the OpenAPI specification: {e}")
        return None


# Example usage

if __name__ == "__main__":
    url = "http://127.0.0.1:8010"
    openapi_spec = fetch_openapi_spec(url)

    if openapi_spec is not None:
        print("Fetched OpenAPI Specification:")
    else:
        print("Failed to fetch OpenAPI Specification.")

    gpt_tool = generate_custom_openapi_spec(
        "https://127.0.0.1",
        openapi_spec,
        "Title",
        "Description",
        ["/execute-tool/", "/find-tools/", "/get_new_events/"],
    )

    print(gpt_tool)
