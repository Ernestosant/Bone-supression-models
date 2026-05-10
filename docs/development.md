# Development

## Environment Setup

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install model-specific requirements only when running real checkpoint inference. The unit tests use
mocked models and do not require TensorFlow or FastAI checkpoints.

## Code Organization

- `src/bone_suppression/preprocessing.py`: image shape, dtype, histogram, resize, and normalization
  helpers.
- `src/bone_suppression/registry.py`: model registry loader and validation.
- `src/bone_suppression/model_io.py`: optional framework checkpoint loading.
- `src/bone_suppression/inference.py`: reusable prediction routines.
- `src/bone_suppression/gradio_app.py`: interactive UI wrapper.
- `src/bone_suppression/cli.py`: command line entrypoint.

## Quality Checks

```bash
ruff check .
pytest
```

The GitHub Actions workflow runs the same checks on pushes and pull requests.

## Adding A Model

1. Add a new entry to `configs/model_registry.json`.
2. Implement or extend the appropriate inference routine.
3. Add tests that use a mocked model object.
4. Document checkpoint requirements and limitations in `docs/models.md`.
5. Keep checkpoint files outside Git.

## Documentation Updates

Update the README when user-facing setup, model availability, or visual examples change. Update
`docs/` when research assumptions, dataset handling, or developer workflow changes.
