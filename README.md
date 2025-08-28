## RAVANA AGI — Autonomous, evolving agentic system

RAVANA is an experimental open-source AGI framework focused on continuous autonomous operation, memory, emotional modeling, and self-improvement. This repository contains the core runtime, modules, services, and documentation required to run, extend, and research the system.

## Quick links

- Code: `core/`, `modules/`, `services/`, `database/`, `main.py`
- Install & requirements: `requirements.txt`, `pyproject.toml`
- Run (example entrypoint): `main.py`
- Docs folder: `docs/`
- Project wiki (detailed design & guides): `wiki/` (key pages below)
- License: `LICENSE`

## Important wiki pages

The `wiki/` directory mirrors in-repo design docs. Start here:

- [Project Overview](wiki/Project%20Overview.md)
- [Architecture & Design](wiki/Architecture%20&%20Design.md)
- [Core System](wiki/Core%20System.md)
- [API Reference](wiki/API%20Reference.md)
- [Development Guide](wiki/Development%20Guide.md)
- [Memory Systems](wiki/Memory%20Systems.md)
- [LLM Integration](wiki/LLM%20Integration.md)
- [Services](wiki/Services.md)

There are many more focused articles inside `wiki/` (Action System, Decision-Making, Emotional Intelligence, etc.) — browse the folder for module-level details.

## Documentation in `docs/`

The `docs/` folder contains user-facing and developer docs, API references, and examples. Use `docs/index.md` as the landing page. API specs live under `docs/api/` and developer notes under `docs/development/`.

## Getting started (local)

1) Clone the repository

```bash
git clone https://github.com/OpenSource-Syndicate/RAVANA.git
cd RAVANA
```

2) Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# or, for editable install during development:
pip install -e .
```

3) Run the main entrypoint (example)

```bash
python main.py
```

Notes:
- Some modules and services are experimental and may require additional configuration files under `core/config.json` or `core/config.py`.
- If you plan to run long experiments, configure the environment and service endpoints as described in `docs/development/` and `wiki/Deployment%20&%20Operations.md`.

## Project layout (short)

- `main.py` — example runtime entrypoint
- `core/` — orchestrator, state manager, agents, and internal services
- `modules/` — pluggable modules (self-reflection, experimentation, conversational ai, etc.)
- `services/` — service-level code (data, memory, knowledge)
- `database/` — DB engine, models and schema
- `docs/` — curated documentation
- `wiki/` — extended design docs and API details (see list above)

## Contributing

See `wiki/Development Guide.md` and `docs/development/` for contribution instructions, coding standards, and the review process. Create feature branches from `main`, add tests where possible, and open pull requests describing changes.

## Tests and helpful scripts

- `run_physics_tests.py` and `scae_benchmark.py` are example harnesses present in the repo. Inspect and run them from an activated virtualenv.

## License

This project is distributed under the MIT License — see `LICENSE`.

## Where to get help

- Read `wiki/Project%20Overview.md` for goals and architecture.
- For developer questions, open an issue and tag it `help wanted`.

---

Requirements coverage:

- Rewrite `README.md` with proper links — Done
- Link in-repo `docs/` and `wiki/` pages mentioned in attachments — Done

If you'd like, I can also:
- Add a short `README` inside `docs/` or `wiki/` that links sub-pages (toc)
- Create a CONTRIBUTING.md that mirrors the `wiki/Development Guide.md`