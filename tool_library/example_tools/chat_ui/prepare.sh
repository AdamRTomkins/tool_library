# Substitute environment variables in _readme.md and output to new_readme.md
#envsubst < _chainlit.md > chainlit.md

#!/bin/sh

# Generate the secret and capture the output
output=$(chainlit create-secret)

# Extract the line containing 'CHAINLIT_AUTH_SECRET=' and write it to the .env file
echo "$output" | grep 'CHAINLIT_AUTH_SECRET=' > .env

echo "The CHAINLIT_AUTH_SECRET has been successfully written to .env"
