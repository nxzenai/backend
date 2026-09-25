"""Transactional access emails using the existing SMTP configuration."""
from html import escape

from services.demo_email import _message, _send


def _deliver(subject: str, recipients: list[str], heading: str, body: str) -> None:
    message = _message(subject, recipients)
    message.set_content(f"{heading}\n\n{body}\n\nRegards,\nThe NxZenAI Team")
    message.add_alternative(
        "<html><body style='font-family:Arial,sans-serif;color:#172033'>"
        "<div style='max-width:620px;margin:24px auto;border:1px solid #e4e9f2;border-radius:12px;overflow:hidden'>"
        "<div style='background:#071426;padding:24px 32px;color:white;font-size:24px;font-weight:700'>NxZenAI</div>"
        f"<div style='padding:32px'><h2>{escape(heading)}</h2><p style='white-space:pre-line'>{escape(body)}</p>"
        "<p>Regards,<br><strong>The NxZenAI Team</strong></p></div></div></body></html>",
        subtype="html",
    )
    _send(message)


def send_access_request_admin(email: str, full_name: str, recipients: list[str]) -> None:
    _deliver(
        "New NxZenAI Studio access request", recipients, "Access approval requested",
        f"{full_name} ({email}) registered and is waiting for approval in User Management.",
    )


def send_access_decision(email: str, full_name: str, approved: bool, reason: str | None = None) -> None:
    state = "approved" if approved else "rejected"
    detail = "You can now sign in to NxZenAI Studio." if approved else (reason or "Contact an administrator for details.")
    _deliver(
        f"Your NxZenAI Studio access was {state}", [email], f"Access request {state}",
        f"Hello {full_name},\n\n{detail}",
    )
