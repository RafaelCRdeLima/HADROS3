# HADROS3 Seminar

A self-contained web presentation of the HADROS3 pipeline, from Kerr-system
visualization to local per-site transport with GEANT4. The final section
discusses modern collider and theoretical validation, the limits of applying
standard GEANT4 matter at torus densities, and the V1--V8 numerical campaign.

## Present locally

Open `index.html` directly or serve the documentation directory:

```bash
python3 -m http.server 8000 --directory docs
```

and open `http://127.0.0.1:8000/seminar/`.

Controls: arrow keys or Space to navigate, `F` for fullscreen, `O` for the
overview, `N` for speaker notes, and `?` for help.

## GitHub Pages

The `pages.yml` workflow publishes `docs/` after every push to `main`. The
presentation is available at:

```text
https://rafaelcrdelima.github.io/HADROS3/seminar/
```

If the repository is not yet configured to deploy Pages with Actions, select
**Settings → Pages → Source → GitHub Actions** once.

## Physics review and references

The full review linked from slide 18 is stored in
`docs/bibliography/HADROS3_UHE_Dense_Matter/`, together with 16 downloaded
papers, BibTeX metadata, and SHA-256 checksums.

## Update snapshots

Files in `assets/` are frozen copies from a validated run. This prevents the
presentation from depending on the local `output/` directory, which is normally
ignored by Git.
