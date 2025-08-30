import re
from playwright.sync_api import sync_playwright, Page, expect

def run_verification(page: Page):
    """
    This script verifies that a user can navigate to the Strategy Manager,
    launch a bot with the MACD strategy, and see a success message.
    """
    # 1. Arrange: Go to the application's homepage.
    # The frontend server runs on port 5173.
    page.goto("http://localhost:5173")

    # 2. Act: Navigate to the Strategy Manager page.
    # We use get_by_role for robust selection.
    strategy_link = page.get_by_role("link", name="Strategies")
    strategy_link.click()

    # 3. Assert: Check if we are on the correct page.
    # The `expect` function will wait for the element to be visible.
    expect(page.get_by_role("heading", name="Strategy Manager")).to_be_visible()

    # 4. Act: Launch the bot using the default MACD parameters.
    launch_button = page.get_by_role("button", name="Launch Bot")
    launch_button.click()

    # 5. Assert: Check for the success feedback message.
    # The message might take a moment to appear after the API call.
    success_message = page.locator(".feedback-message.success")
    expect(success_message).to_be_visible()
    expect(success_message).to_contain_text(re.compile("Bot .* started successfully!|Signal bot .* start process initiated.", re.IGNORECASE))

    # 6. Screenshot: Capture the final state for visual verification.
    page.screenshot(path="jules-scratch/verification/verification.png")

# --- Playwright Boilerplate ---
def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        run_verification(page)
        browser.close()

if __name__ == "__main__":
    main()
