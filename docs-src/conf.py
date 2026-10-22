# Sphinx configuration of the documentation (docs-src -> docs/documentation).
#   sphinx-build -b dirhtml docs-src docs/documentation
project = "Mocap to MSK"
author = "Mocap to MSK contributors"
copyright = "2026, Mocap to MSK contributors"
extensions = ["myst_parser", "sphinx_design", "sphinx_copybutton"]
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "attrs_inline", "attrs_block"]
myst_heading_anchors = 3
source_suffix = {".md": "markdown"}
exclude_patterns = ["_build", "README.md", "_generated"]
html_theme = "sphinx_book_theme"
html_title = "Mocap to MSK documentation"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["redirects.js"]
html_favicon = "_static/favicon.svg"
html_last_updated_fmt = "%b %d, %Y"
templates_path = ["_templates"]
html_sidebars = {"**": ["navbar-logo.html", "sidebar-links.html", "search-field.html", "sbt-sidebar-nav.html"]}
html_theme_options = {
  "logo": {"text": "Mocap to MSK", "image_light": "_static/logo.svg", "image_dark": "_static/logo-dark.svg"},
  "repository_url": "https://github.com/Fitz-Yuan/mocap-to-myofullbody",
  "repository_branch": "main",
  "path_to_docs": "docs-src",
  "use_repository_button": False,
  "use_issues_button": False,
  "use_edit_page_button": False,
  "use_download_button": False,
  "use_fullscreen_button": False,
  "home_page_in_toc": True,
  "show_toc_level": 2,
  "navigation_with_keys": False,
  "article_header_end": [],
  "icon_links": [],
}
