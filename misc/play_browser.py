import argparse
import asyncio
from playwright.async_api import async_playwright, Page
import twentyfortyeight as twfe
import numpy as np
import math
import train

# Instructions:
# 1. Clone the original 2048: https://github.com/gabrielecirulli/2048 (all versions hosted on the web have way too many ads)
# 2. Host it locally using python3 -m http.server
# 3. Run this script as python3 -m misc.play_browser http://localhost:8080

# N.B. We could make this into a gym environment, but I don't feel like it right now.
async def state_from_page(page: Page):
    over = await page.evaluate("window.gameManager.over")
    grid = await page.evaluate("window.gameManager.grid")
    cells = grid["cells"]

    tiles = np.zeros((4, 4), dtype=np.uint8)

    for y in range(4):
        for x in range(4):
            cell = cells[x][y]
            if cell is None:
                continue

            tiles[y, x] = math.log2(cell["value"])
    
    return twfe.State.from_tiles(tiles), over

async def play(page: Page, agent: train.TwentyfortyEightAgent):
    await page.evaluate("window.gameManager.keepPlaying = true")

    state, _over = await state_from_page(page)

    while not state.is_terminated:
        action = agent.model.get_actions(state.tiles, state.action_mask).item()

        # Convert action to what the page expect
        if action == twfe.UP:
            act = 0
        elif action == twfe.RIGHT:
            act = 1
        elif action == twfe.DOWN:
            act = 2
        elif action == twfe.LEFT:
            act = 3

        await page.evaluate(f"window.gameManager.move({act})")
        await asyncio.sleep(0.05)

        state, _over = await state_from_page(page)
    
    print("Done")

async def wait_forever():
    event = asyncio.Event()
    await event.wait()

async def main(url, agent: train.TwentyfortyEightAgent):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, args=["--start-maximized"])
        page = await browser.new_page(no_viewport=True)

        # Bit of javascript to extract a reference to the GameManager object lol
        await page.add_init_script("""
            Object.defineProperty(Object.prototype, 'startTiles', {
                get() { return this._startTiles; },
                set(value) { this._startTiles = value; window.gameManager = this; }
            });
        """)
        await page.goto(url)

        # Bit hacky (we need to wait for the initialization code to run, and we do not exactly control the
        # timing of that)
        await asyncio.sleep(1)
        await play(page, agent)

        await wait_forever()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("model_pth")
    args = parser.parse_args()

    # Bit hacky to load it like this but what can you do
    agent = train.TwentyfortyEightAgent()
    agent.load_checkpoint_dict(args.model_pth)

    asyncio.run(main(args.url, agent))