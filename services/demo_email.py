from datetime import date
from email.message import EmailMessage
from email.utils import formataddr
from html import escape
import smtplib
import ssl

from app.core.config.settings import settings


def _formatted_demo_date(value: str) -> str:
    selected_date = date.fromisoformat(value)
    return selected_date.strftime("%A, %B %d, %Y")


def _message(subject: str, recipients: list[str]) -> EmailMessage:
    from_email = settings.smtp_from_email or settings.smtp_username
    if not settings.smtp_host or not from_email:
        raise RuntimeError("SMTP is not configured")
    if not recipients:
        raise RuntimeError("SMTP recipients are not configured")
    if bool(settings.smtp_username) != bool(settings.smtp_password):
        raise RuntimeError("SMTP credentials are incomplete")
    if settings.smtp_use_tls and settings.smtp_use_ssl:
        raise RuntimeError("SMTP TLS and SSL cannot both be enabled")

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = formataddr((settings.smtp_from_name, from_email))
    message["To"] = ", ".join(recipients)
    return message


def _send(message: EmailMessage) -> None:
    context = ssl.create_default_context()
    if settings.smtp_use_ssl:
        smtp = smtplib.SMTP_SSL(
            settings.smtp_host,
            settings.smtp_port,
            timeout=20,
            context=context,
        )
    else:
        smtp = smtplib.SMTP(
            settings.smtp_host,
            settings.smtp_port,
            timeout=20,
        )

    with smtp:
        if settings.smtp_use_tls:
            smtp.starttls(context=context)
        if settings.smtp_username and settings.smtp_password:
            smtp.login(settings.smtp_username, settings.smtp_password)
        smtp.send_message(message)


def send_customer_demo_confirmation(lead: dict) -> None:
    name = " ".join(str(lead["name"]).split())[:100]
    demo_date = _formatted_demo_date(str(lead["preferred_demo_date"]))
    message = _message(
        "Your NxZenAI Demo Is Scheduled",
        [str(lead["email"])],
    )
    message.set_content(
        f"Hello {name},\n\n"
        "Thank you for requesting an NxZenAI demo. We have received your "
        f"request for {demo_date}.\n\n"
        "Our team will contact you with any confirmed session details.\n\n"
        "Regards,\nThe NxZenAI Team"
    )
    message.add_alternative(
        f"""
        <html><body style="margin:0;background:#f4f7fb;font-family:Arial,sans-serif;color:#172033">
          <div style="max-width:620px;margin:24px auto;background:#ffffff;border-radius:12px;overflow:hidden;border:1px solid #e4e9f2">
            <div style="background:#071426;padding:24px 32px;color:#ffffff">
              <div style="font-size:24px;font-weight:700">NxZenAI</div>
              <div style="color:#91b8ff;margin-top:4px">Demo request confirmed</div>
            </div>
            <div style="padding:32px">
              <p>Hello {escape(name)},</p>
              <p>Thank you for requesting an NxZenAI demo. We have received your request.</p>
              <div style="margin:24px 0;padding:18px;border-left:4px solid #2563eb;background:#f0f5ff">
                <div style="font-size:12px;text-transform:uppercase;color:#5c6780">Selected Saturday</div>
                <div style="font-size:18px;font-weight:700;margin-top:6px">{escape(demo_date)}</div>
              </div>
              <p>Our team will contact you with any confirmed session details.</p>
              <p style="margin-top:28px">Regards,<br><strong>The NxZenAI Team</strong></p>
            </div>
          </div>
        </body></html>
        """,
        subtype="html",
    )
    _send(message)


def send_admin_demo_notification(lead: dict) -> None:
    name = " ".join(str(lead["name"]).split())[:100]
    demo_date = _formatted_demo_date(str(lead["preferred_demo_date"]))
    message = _message(
        f"New NxZenAI Demo Booking \N{EN DASH} {name}",
        settings.smtp_admin_recipients,
    )

    fields = [
        ("Name", name),
        ("Email", str(lead["email"])),
        ("Phone", str(lead.get("phone") or "Not provided")),
        ("Profession", str(lead.get("profession") or "Not provided")),
        ("Program interest", str(lead.get("program_interest") or "Not provided")),
        ("Selected Saturday", demo_date),
        ("Message", str(lead.get("message") or "Not provided")),
    ]
    message.set_content(
        "A new NxZenAI demo request was submitted.\n\n"
        + "\n".join(f"{label}: {value}" for label, value in fields)
    )
    rows = "".join(
        "<tr>"
        f'<th style="padding:10px;text-align:left;vertical-align:top;background:#f4f7fb">{escape(label)}</th>'
        f'<td style="padding:10px;white-space:pre-wrap">{escape(value)}</td>'
        "</tr>"
        for label, value in fields
    )
    message.add_alternative(
        f"""
        <html><body style="font-family:Arial,sans-serif;color:#172033">
          <h2>New NxZenAI Demo Booking</h2>
          <table style="border-collapse:collapse;width:100%;max-width:720px;border:1px solid #dfe5ef">{rows}</table>
        </body></html>
        """,
        subtype="html",
    )
    _send(message)
