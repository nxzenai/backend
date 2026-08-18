# Python Lab Phase 1 Private-Staging Smoke Test

Run this checklist only in an isolated private-staging environment with test users and a test database. Do not run it against production. Record request IDs and timestamps; never paste tokens or credentials into the record.

## Preconditions

- Confirm the previously exposed MongoDB credential has been revoked and replaced through the deployment secret store.
- Review repository history and deployment logs for use of the retired credential.
- Set `ENVIRONMENT=staging`, `APP_DEBUG=false`, a unique strong `SECRET_KEY`, an explicit staging `MONGODB_URL`, TLS-enabled object storage, exact HTTPS `CORS_ORIGINS`, and an HTTPS `NEXT_PUBLIC_STUDIO_API_URL`.
- Restrict access to trusted staging users. Python still executes on the API host.
- Use two disposable accounts, User A and User B, and a disposable notebook.

## Checklist

1. In a signed-out browser, open `/notebooks/<test-id>` and verify the login redirect occurs without notebook content flashing.
2. Sign in as User A and create a private notebook.
3. Create a code cell containing:

   ```python
   value = 21
   print(value * 2)
   ```

4. Run it and verify output `42`; verify cell and notebook execution counts advance.
5. Create and run a second code cell containing `print(value)`; verify output `21`, proving kernel persistence.
6. Refresh the browser and verify both outputs and counts remain persisted.
7. Restart the kernel, rerun `print(value)`, and verify a controlled `NameError` output appears.
8. Start this bounded cell, then interrupt it within two seconds:

   ```python
   import time
   for _ in range(40):
       time.sleep(0.25)
   ```

9. Verify interrupt returns promptly, the UI stays responsive, and kernel status leaves `busy`.
10. Clear one cell output and verify no other cell changes.
11. Clear all outputs and verify execution counts follow the documented preservation rule.
12. Exercise status, restart, shutdown, repeated shutdown, and starting a new execution after shutdown. Verify visible lifecycle changes and controlled messages.
13. Enter a new draft and immediately use Run and Shift+Enter. Verify only the latest text executes.
14. Edit multiple cells and immediately use Run All. Verify current drafts execute in notebook order and markdown is skipped.
15. Temporarily force a save failure in staging and verify Run/Run All is blocked with an accessible visible error.
16. Render malicious HTML containing script, event-handler, `javascript:` URL, iframe, and CSS URL payloads. Verify none execute and the output iframe has no script permission.
17. Export HTML and verify malicious title, description, source, and output text is escaped and readable when opened offline.
18. Export `.ipynb`, open it with Jupyter, and validate it with `nbformat.validate`.
19. As User B, try fetch, edit, delete, cell CRUD/reorder, execute, execute-all, clear, interrupt, restart, shutdown, and status against User A's notebook. Every operation must return the non-disclosing `404` response.
20. Sign out and repeat all eight execution/lifecycle endpoints without a token. Every request must return `401` and cause no mutation or kernel action.
21. Inspect staging application logs for the test window. Confirm they contain no bearer tokens, database URIs, credentials, environment values, absolute server paths, or raw tracebacks.

## Cleanup

- Shut down the test kernel.
- Delete the disposable notebook and users through approved staging administration paths.
- Remove any intentionally injected failure configuration.
- Attach sanitized evidence to the deployment record.

Any failed item blocks staging promotion. Public multi-user production remains out of scope until execution is isolated with enforced resource controls.
