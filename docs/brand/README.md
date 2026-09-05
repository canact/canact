# Brand source

`canact.svg` is the mark. PNG and other rasters are generated from it.

```bash
make brand
```

That writes `/tmp/canact-brand/org-avatar-1024.png` and
`/tmp/canact-brand/social-preview.png`. Do not commit those files.
GitHub org avatar and repo social preview are uploaded from the
generated PNG. Edit the SVG, then re-run `make brand`.
