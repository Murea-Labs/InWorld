# Website files

The public website is served from the repository root by GitHub Pages.
The root `index.html` is the entry point; all page assets live here.

- `assets/css/`, `assets/js/`, `assets/fonts/`, `assets/vendor/`: presentation and browser code.
- `assets/img/`: site illustrations, figures, and video posters.
- `assets/video/`: local preview videos, story ordering, and review records.
- `paper/`: the downloadable technical report.
- `favicon.ico` and `sitemap.xml`: site icon and search index.

Keep `CNAME` at the repository root for the existing GitHub Pages custom domain.
Keep `robots.txt` at the root; its sitemap entry points to `page/sitemap.xml`.
Model code can live alongside `index.html` and `page/` without mixing with website assets.

Serve the repository root for local previews: `python3 -m http.server 8000`.
Open `http://localhost:8000/`, not the `page/` directory.
HTML resource paths start with `page/`. CSS font and image paths remain relative
to their stylesheets. Hosted story video URLs stay external; local poster paths
in `assets/video/story-order.json` are relative to the root entry point.
