# defog_utils

Internal utilities library for Defog. These utilities comprise:
* sql/schema/instruction feature extraction
* database connectors and validators

These utilities should be purely stateless and have minimal library dependencies.

## Installation

```bash
pip install -e .
```
We recommend using `-e` flag to install in editable mode, since we foresee frequent updates to this library.

## Tests

```bash
pytest
```