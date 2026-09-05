# Documentation website

The English and Chinese READMEs remain the source of truth. The MkDocs hook
splits them into 21 chapters per language, preserves code examples, copies only
referenced images, and resolves repository links. Generated pages are ignored
by Git. Training and inference code is not involved in the website build.

## Preview locally

Use a separate documentation environment; no GPU or CUDA packages are needed.

```bash
python -m venv .venv-docs
source .venv-docs/bin/activate
python -m pip install -r requirements-docs.txt
python -m mkdocs serve
```

On Windows, activate with `.venv-docs\Scripts\activate` instead. This refers to
the documentation build only, not the training environment.

Build and validate the static output:

```bash
python -m mkdocs build --strict
python docs/check_site.py
```

The English site is in `site/` and Chinese pages are in `site/zh/`. Search,
language switching, light/dark themes, chapter navigation and code-copy buttons
are supplied by Material for MkDocs and mkdocs-static-i18n.

## Publish on GitHub Pages

1. Review and merge the documentation pull request.
2. In the repository, open **Settings → Pages → Build and deployment** and
   choose **GitHub Actions** as the source.
3. Run **Actions → Documentation → Run workflow** on `main` if the initial
   deployment ran before Pages was enabled.

Expected address after a successful deployment:
<https://wangyulin-seu.github.io/HCCEPose/>.

Pull requests build and upload the site artifact but never deploy it. Changes
to the READMEs, documentation configuration, or referenced image directories
trigger a rebuild on `main`. The deploy job has the Pages and OIDC permissions;
the build job has only read access to repository contents.

## Maintenance

Edit `README.md` and `README_CN.md` to change tutorial content. If you rename or
reorder a chapter boundary, update `SECTIONS` in `docs/hooks.py` as well. Missing
boundaries, local links, images, or code-fence terminators fail the build instead
of silently losing content. Do not edit `docs/content/` directly.

The website uses the repository's existing figures and GIFs, loaded lazily. It
labels the old roadmap and leaderboard images as historical. References to the
absent `hf-dataset-card/README*.md` files link to the actual Hugging Face dataset
card. The two source READMEs themselves are preserved unchanged.

Official deployment reference:
<https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages>.
