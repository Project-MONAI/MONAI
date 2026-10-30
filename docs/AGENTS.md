# Documentation

Sources live in `docs/source`: one page per area (`transforms.rst`,
`networks.rst`, `data.rst`, `losses.rst`, `metrics.rst`, `engines.rst`,
`bundle.rst`, and the rest), long-form guides in Markdown, and the Sphinx
configuration in `conf.py`.

## Things that are easy to get wrong

- The per-area `.rst` pages are the public API lists. A new public class or
  function that is not added to its area page does not appear in the
  published documentation, even if it is exported in `__init__.py`.
- Build before submitting documentation changes:

  ```bash
  pip install -r docs/requirements.txt
  cd docs && make html
  ```

  and check the page you changed in `docs/build/html`. Do not commit the
  build output.
- Python docstrings are Google style and written in reStructuredText; they
  are pulled into these pages, so a docstring error is a documentation
  build error.
- Release notes and feature summaries go in the `whatsnew` page for the
  upcoming release, not into older `whatsnew_*.md` files.
- Spelling is American English, matching the code (see `CONTRIBUTING.md`).
