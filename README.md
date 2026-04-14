# Slice Stacker

Focus stacking tools for macro photography.

## Installation

```bash
pip install -e .
```

## Usage

### Focus Stack

Stack images taken at different focal planes (e.g., using in-camera focus bracketing):

```bash
focus-stack image1.jpg image2.jpg image3.jpg -o output.jpg
```

### Rail Stack

Stack images from a macro rail sequence (coming soon):

```bash
rail-stack image1.jpg image2.jpg image3.jpg -o output.jpg
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
