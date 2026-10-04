# Honours project webpage

A self-contained static webpage presenting **Experience-Based Belief Learning for Structured Animation Code Generation**.

## Preview locally

From this directory, run:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

The page has no external framework or font dependency. All figures, the example video and the seminar-slide download are stored in `assets/`.

The embedded example is a 20.6-second, 1080p/30 fps DP-3T section configured to loop continuously.

## Edit

- Content and structure: `index.html`
- Visual design and responsive layout: `styles.css`
- Mobile navigation and result-number animation: `script.js`

## Deploy

Upload the complete folder to any static host, such as GitHub Pages, Netlify, an institutional web server or an LMS web-content area. Keep the relative file structure unchanged.
