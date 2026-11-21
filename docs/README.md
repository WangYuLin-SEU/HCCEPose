# Website sources

Run `python -m pip install -r docs/requirements.txt`, then `python docs/build.py` from this package.
The builder validates local references and recreates only its `pages/` output directory.
Use `python -m http.server 8768 --directory pages` for a local preview.

Edit `docs/zh/` and `docs/en/` together. Chinese examples use Chinese comments; English examples use English comments.
Media and third-party licenses are listed in the project’s `PUBLIC_SOURCES.md`.

The navigation covers getting started, image/video examples, WAPR/MegaPose, acceleration, new-object preparation and training, APIs, BOP evaluation and public history. Stage-script tutorials link to a fixed public upstream revision and identify where commands run. Keep application entry points and upstream scripts distinct when updating paths.
