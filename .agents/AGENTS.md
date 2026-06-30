# Agent Rules for SDGFT Repository

## Git & Repository Maintenance
- **Papers Directory:** Do NOT commit non-PDF files in the `papers/` directory (e.g. `.tex`, `.bib`, `.sty`, or build scripts). Only compiled `.pdf` files should be tracked by Git.
- **Lock Files:** Do NOT commit `uv.lock` or similar virtual environment lockfiles to the repository unless explicitly requested by the user.
- **Data Files:** Do NOT commit large external dataset files (e.g., SPARC `.dat` files from `data/sparc/Rotmod_LTG/`). Ensure these stay in `.gitignore`.
