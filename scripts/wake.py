"""Keep the Streamlit Community Cloud app awake.

Free apps sleep after a period without visitors and only wake when someone clicks the
"get this app back up" button, so a plain HTTP ping is not enough. This opens the page in
headless Chromium and clicks the button if it is there. Run by .github/workflows/keep-awake.yml.
"""
from __future__ import annotations

import asyncio
import os
import sys


async def main(url: str) -> None:
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=120_000)
        await page.wait_for_timeout(8_000)
        button = page.get_by_role("button", name="Yes, get this app back up!")
        if await button.count():
            await button.click()
            await page.wait_for_timeout(60_000)
            print(f"WOKE {url}")
        else:
            print(f"AWAKE {url}")
        await browser.close()


if __name__ == "__main__":
    target = os.environ.get("APP_URL") or (sys.argv[1] if len(sys.argv) > 1 else "")
    if not target:
        raise SystemExit("Set APP_URL (repo variable) or pass the app URL as an argument.")
    asyncio.run(main(target))
