# Validation and API spending

- Do not run tests unless the user explicitly requests them. A general request to review or double-check work is not permission to run tests.
- Do not run `earnings.py`, live API smoke tests, agent sessions, or other checks that spend API tokens or incur charges without explicit user approval for that run.
- Default to static code and configuration review. Explain any checks that were not run; do not claim runtime verification.
- The local unit tests use mocked responses, but still require an explicit user request before running them.
