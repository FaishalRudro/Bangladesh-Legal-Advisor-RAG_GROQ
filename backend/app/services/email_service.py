import aiosmtplib
from email.message import EmailMessage
from email.utils import formataddr
import random
import string
from datetime import datetime
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

def generate_otp(length=6):
    return ''.join(random.choices(string.digits, k=length))

def get_html_template(title: str, content_html: str) -> str:
    year = datetime.now().year
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #f4f7f6;
                color: #333;
                margin: 0;
                padding: 0;
            }}
            .container {{
                max-width: 600px;
                margin: 40px auto;
                background-color: #ffffff;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
                border-top: 5px solid #046e4b;
            }}
            .header {{
                text-align: center;
                padding: 30px 20px;
                background-color: #046e4b;
                color: #ffffff;
            }}
            .opening_statement {{
                font-size: 28px;
                font-family: 'Amiri', 'Traditional Legal', serif;
                margin-bottom: 10px;
                color: #e6b800;
            }}
            .title {{
                font-size: 22px;
                font-weight: 600;
                margin: 0;
                letter-spacing: 1px;
            }}
            .content {{
                padding: 40px 30px;
                line-height: 1.6;
                font-size: 16px;
                color: #4a4a4a;
            }}
            .greeting {{
                font-size: 18px;
                font-weight: 500;
                color: #046e4b;
                margin-bottom: 20px;
            }}
            .otp-box {{
                background-color: #f0fdf4;
                border: 1px dashed #046e4b;
                text-align: center;
                padding: 15px;
                margin: 30px 0;
                border-radius: 5px;
                font-size: 32px;
                font-weight: bold;
                letter-spacing: 5px;
                color: #046e4b;
            }}
            .footer {{
                background-color: #f4f7f6;
                text-align: center;
                padding: 20px;
                font-size: 13px;
                color: #888;
                border-top: 1px solid #eaeaea;
            }}
            .jazakallah {{
                font-style: italic;
                margin-bottom: 10px;
                color: #046e4b;
            }}
            .button {{
                display: inline-block;
                padding: 12px 25px;
                background-color: #046e4b;
                color: #ffffff;
                text-decoration: none;
                border-radius: 4px;
                font-weight: bold;
                margin-top: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="opening_statement">بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ</div>
                <h1 class="title">{title}</h1>
            </div>
            <div class="content">
                <div class="greeting">Dear User wa Rahmatullah,</div>
                {content_html}
                <div style="margin-top: 40px;">
                    <p class="jazakallah">Best regards,</p>
                    <p style="margin: 4px 0 0 0;font-size: 16px; color: #777;"><strong>The Bangladesh Legal Advisor Team</strong></p>
                    <p style="margin: 4px 0 0 0; font-size: 13px; color: #777;">As-Sunnah Foundation</p>
                </div>
            </div>
            <div class="footer">
                <p>&copy; {year} Bangladesh Legal Advisor. All rights reserved.</p>
                <p>This is an automated message. Please do not reply directly to this email.</p>
            </div>
        </div>
    </body>
    </html>
    """

async def send_email(to_email: str, subject: str, html_content: str, text_content: str):
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = formataddr(("Bangladesh Legal Advisor", settings.FROM_EMAIL))
    msg['To'] = to_email
    
    msg.set_content(text_content)
    msg.add_alternative(html_content, subtype='html')

    try:
        await aiosmtplib.send(
            msg,
            hostname=settings.SMTP_HOST,
            port=settings.SMTP_PORT,
            start_tls=True,
            username=settings.SMTP_USER,
            password=settings.SMTP_PASSWORD
        )
        logger.info(f"Email sent successfully to {to_email}")
    except Exception as e:
        logger.error(f"Failed to send email to {to_email}: {e}")

async def send_otp_email(to_email: str, otp: str):
    subject = "Verify your account - Bangladesh Legal Advisor"
    
    text_content = f"Dear User,\n\nYour OTP for verification is: {otp}\n\nDo not share this code with anyone.\n\nJazakallah Khair,\nThe Bangladesh Legal Advisor Team"
    
    html_content = get_html_template(
        "Account Verification",
        f"""
        <p>Thank you for registering with Bangladesh Legal Advisor. Please use the verification code below to verify your account.</p>
        <div class="otp-box">{otp}</div>
        <p>If you did not request this code, please ignore this email.</p>
        """
    )
    
    await send_email(to_email, subject, html_content, text_content)

async def send_invitation_email(to_email: str):
    subject = "Lawyer Invitation - Bangladesh Legal Advisor"
    
    invite_link = f"{settings.FRONTEND_URL}/accept-invite?email={to_email}"
    
    text_content = f"Dear User,\n\nYou have been invited as a Lawyer to Bangladesh Legal Advisor.\n\nPlease accept the invitation and set your password to login by visiting the following link:\n{invite_link}\n\nJazakallah Khair,\nThe Bangladesh Legal Advisor Team"
    
    html_content = get_html_template(
        "Lawyer Invitation",
        f"""
        <p>You have been formally invited to join the <strong>Bangladesh Legal Advisor</strong> platform as an authorized <strong>Lawyer</strong>.</p>
        <p>Your expertise is highly valued. Please proceed to the platform to accept your invitation and establish your secure credentials.</p>
        <center>
            <a href="{invite_link}" class="button" style="color: #ffffff;">Accept Invitation</a>
        </center>
        """
    )
    
    await send_email(to_email, subject, html_content, text_content)

async def send_password_change_otp_email(to_email: str, otp: str):
    subject = "Password Change OTP - Bangladesh Legal Advisor"
    text_content = f"Dear User,\n\nYour OTP for changing your password is: {otp}\n\nDo not share this code with anyone.\n\nJazakallah Khair,\nThe Bangladesh Legal Advisor Team"
    html_content = get_html_template(
        "Password Change Request",
        f"""
        <p>You have requested to change your password for Bangladesh Legal Advisor. Please use the verification code below to authorize this change.</p>
        <div class="otp-box">{otp}</div>
        <p>If you did not request this code, please secure your account immediately and ignore this email.</p>
        """
    )
    await send_email(to_email, subject, html_content, text_content)

async def send_reset_password_email(to_email: str, reset_token: str):
    subject = "Reset Your Password - Bangladesh Legal Advisor"
    reset_link = f"{settings.FRONTEND_URL}/reset-password?token={reset_token}"
    text_content = f"Dear User,\n\nYou requested to reset your password.\n\nPlease visit the following link to reset your password:\n{reset_link}\n\nIf you did not request this, please ignore this email.\n\nJazakallah Khair,\nThe Bangladesh Legal Advisor Team"
    html_content = get_html_template(
        "Password Reset Request",
        f"""
        <p>We received a request to reset the password for your Bangladesh Legal Advisor account.</p>
        <p>Please click the button below to securely reset your password. This link will expire in 1 hour.</p>
        <center>
            <a href="{reset_link}" class="button" style="color: #ffffff;">Reset Password</a>
        </center>
        <p>If you did not request a password reset, you can safely ignore this email.</p>
        """
    )
    await send_email(to_email, subject, html_content, text_content)
