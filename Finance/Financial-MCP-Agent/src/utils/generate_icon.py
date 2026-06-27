"""Generate custom icon for the A-stock Investment Advisor Agent."""
from PIL import Image, ImageDraw, ImageFont
import os

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "app_icon.ico")


def draw_icon(size):
    """Draw one resolution layer of the icon. Size x Size pixels."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # ── Background: rounded-square-ish deep red gradient ──
    margin = max(1, size // 16)
    r = size // 6  # larger corner radius for cleaner look
    bg_rect = [margin, margin, size - margin, size - margin]

    # Deep red gradient (top → bottom) - richer, more saturated
    for y in range(margin, size - margin):
        t = (y - margin) / max(1, (size - 2 * margin))
        # Top: vibrant crimson #E63946, Bottom: deep blood red #8B0000
        r_val = int(230 - 90 * t)
        g_val = int(57 - 57 * t)
        b_val = int(70 - 70 * t)
        draw.rectangle([margin, y, size - margin, y + 1], fill=(r_val, g_val, b_val, 255))

    # ── Removed gray grid lines for cleaner look ──

    # ── Candlesticks (K-lines) — bigger and bolder for clarity ──
    chart_left = int(size * 0.12)  # more margin on left
    chart_right = int(size * 0.72)  # less right margin to give space for arrow
    chart_bottom = int(size * 0.85)
    chart_top = int(size * 0.15)
    chart_height = chart_bottom - chart_top
    n_candles = 5
    candle_spacing = (chart_right - chart_left) // n_candles
    candle_width = max(4, candle_spacing // 2)  # wider candles for clarity
    wick_width = max(1, size // 64)  # thicker wicks

    # Candlestick data: (open_norm, close_norm, high_norm, low_norm)
    # 0=bottom (chart_bottom), 1=top (chart_top)
    # Rising trend: candles get higher
    candles = [
        (0.75, 0.60, 0.55, 0.78),   # 阳线 - rising
        (0.65, 0.48, 0.42, 0.70),   # 阳线 - rising
        (0.55, 0.45, 0.38, 0.60),   # 阳线 (small body)
        (0.52, 0.30, 0.25, 0.58),   # 阳线 - big surge
        (0.35, 0.12, 0.08, 0.40),   # 大涨阳线!
    ]

    for i, (o, c, h, l) in enumerate(candles):
        cx = chart_left + int(candle_spacing * (i + 0.5))
        body_top = chart_top + int(chart_height * min(o, c))
        body_bot = chart_top + int(chart_height * max(o, c))

        # All bullish (red in China = up, yang line) - brighter, more saturated
        body_color = (255, 70, 50, 255)  # vivid red for bodies
        wick_color = (255, 200, 180, 255)  # brighter wicks for visibility

        # Wick: high (h) → top of wick (small y), low (l) → bottom of wick (large y)
        wick_top = chart_top + int(chart_height * h)
        wick_bot = chart_top + int(chart_height * l)
        draw.rectangle([cx - wick_width, wick_top, cx + wick_width, wick_bot],
                       fill=wick_color)
        # Body
        draw.rectangle([cx - candle_width // 2, body_top,
                        cx + candle_width // 2, body_bot],
                       fill=body_color)
        # Body highlight (left side lighter for 3D effect)
        hl_w = max(1, candle_width // 4)
        draw.rectangle([cx - candle_width // 2, body_top,
                        cx - candle_width // 2 + hl_w, body_bot],
                       fill=(255, 110, 90, 220))

    # ── Upward trend arrow — more prominent, better positioned ──
    arrow_x = int(size * 0.82)  # centered in right area
    arrow_bottom = int(size * 0.75)
    arrow_top = int(size * 0.18)
    arrow_head_height = int(size * 0.15)
    arrow_head_width = int(size * 0.18)
    shaft_w = max(3, size // 32)

    # Arrow shaft
    draw.rectangle([arrow_x - shaft_w // 2, arrow_top + arrow_head_height,
                    arrow_x + shaft_w // 2, arrow_bottom],
                   fill=(255, 230, 80, 255))  # bright gold

    # Arrow head (triangle)
    draw.polygon([
        (arrow_x - arrow_head_width // 2, arrow_top + arrow_head_height),
        (arrow_x + arrow_head_width // 2, arrow_top + arrow_head_height),
        (arrow_x, arrow_top),
    ], fill=(255, 230, 80, 255))

    # ── "牛" character — larger, more centered, with better contrast ──
    try:
        font_size = max(10, size // 5)  # larger font
        font = ImageFont.truetype("C:\\Windows\\Fonts\\simhei.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("C:\\Windows\\Fonts\\msyh.ttc", font_size)
        except Exception:
            font = ImageFont.load_default()

    text = "牛"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    # Position: top-right area, better centered
    tx = int(size * 0.82) - tw // 2
    ty = int(size * 0.25)
    # Shadow for depth
    draw.text((tx + 1, ty + 1), text, fill=(180, 140, 0, 255), font=font)
    # Main text: bright gold
    draw.text((tx, ty), text, fill=(255, 230, 80, 255), font=font)

    # ── Rounded corners overlay (mask corners) ──
    corner_mask = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    mask_draw = ImageDraw.Draw(corner_mask)
    # Fill center
    mask_draw.rounded_rectangle(bg_rect, radius=r, fill=(0, 0, 0, 255))
    # Composite: keep only rounded area
    img = Image.composite(img, Image.new("RGBA", (size, size), (0, 0, 0, 0)), corner_mask)

    return img


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    sizes = [16, 24, 32, 48, 64, 128, 256]
    images = [draw_icon(s) for s in sizes]

    # Save as .ico — largest image first, rest appended
    images[-1].save(
        OUTPUT_PATH,
        format="ICO",
        append_images=list(reversed(images[:-1])),
    )
    print(f"Icon saved to: {OUTPUT_PATH}")
    print(f"Sizes included: {sizes}")


if __name__ == "__main__":
    main()
