# Security policy

## Supported deployment

The current release is intended for one operator on a trusted local machine. Do not expose the API directly to an untrusted network. The local server has no user authentication or tenant isolation. The Docker image listens inside the container; publish it only to loopback or place it behind a reviewed authentication and network boundary.

## Data handling

- Model files are loaded from a local directory with remote downloads and custom model code disabled.
- Prompts and model outputs are processed in memory and are not written to a database by this application.
- The browser sends prompts to the same-origin local API. Do not submit sensitive material unless your machine and deployment are approved for it.
- Capturing hidden states and attention increases resource use. Apply operational limits before shared deployment.

## Reporting a vulnerability

Please do not open a public issue with secrets, private prompts, or exploit details. Contact the repository owner privately through GitHub's security advisory feature. Include a reproducible report and affected commit when possible.
