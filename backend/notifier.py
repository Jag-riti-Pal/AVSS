"""
Module: notifier.py
Purpose: Asynchronous event telemetry and alert notification dispatcher for Telegram & Discord.
"""
from datetime import datetime
import os
from typing import Optional
import httpx


class AlertDispatcher:
    def __init__(
        self,
        discord_webhook_url: Optional[str] = None,
        telegram_bot_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None
    ):
        self.discord_webhook = discord_webhook_url
        self.tg_token = telegram_bot_token
        self.tg_chat_id = telegram_chat_id

    async def send_stranger_alert(
        self,
        event_name: str,
        clip_path: Optional[str] = None,
        snapshot_path: Optional[str] = None
    ) -> None:
        """Dispatches alert notifications to configured webhook channels."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message_body = (
            f"🚨 **SECURITY ALERT: {event_name} DETECTED**\n"
            f"🕒 Time: `{timestamp}`\n"
            f"📹 Clip Stored: `{clip_path if clip_path else 'None'}`"
        )

        print(f"\n[ALERT NOTIFICATION] {timestamp} -> {event_name} (Clip: {clip_path})")

        # 1. Discord Webhook Push
        if self.discord_webhook:
            try:
                async with httpx.AsyncClient(timeout=6.0) as client:
                    await client.post(
                        self.discord_webhook,
                        json={"content": message_body}
                    )
            except Exception as exc:
                print(f"[DISCORD DISPATCH FAILED]: {exc}")

        # 2. Telegram Bot Push (with optional photo snapshot attachment)
        if self.tg_token and self.tg_chat_id:
            try:
                async with httpx.AsyncClient(timeout=8.0) as client:
                    if snapshot_path and os.path.exists(snapshot_path):
                        photo_url = f"https://api.telegram.org/bot{self.tg_token}/sendPhoto"
                        with open(snapshot_path, "rb") as photo_file:
                            files = {"photo": photo_file}
                            data = {"chat_id": self.tg_chat_id, "caption": message_body}
                            await client.post(photo_url, data=data, files=files)
                    else:
                        text_url = f"https://api.telegram.org/bot{self.tg_token}/sendMessage"
                        await client.post(
                            text_url,
                            json={"chat_id": self.tg_chat_id, "text": message_body, "parse_mode": "Markdown"}
                        )
            except Exception as exc:
                print(f"[TELEGRAM DISPATCH FAILED]: {exc}")