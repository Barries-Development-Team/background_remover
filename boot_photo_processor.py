"""
BSS Boot Photo Processor
========================
Watches a folder (your Google Drive sync folder) for new photos.
For each new photo it:
  1. Removes the background (AI, runs locally via rembg)
  2. Applies mild auto color correction (white balance + contrast + slight pop)
  3. Composites the boot onto a backdrop:
       - "white"        : clean white canvas, centered
       - "studio_black" : dark wall + floor with a soft cove fold and a
                          drop shadow, so the boot looks like it's standing
                          on the floor in front of a wall
  4. Saves a finished JPEG to the output folder
  5. Moves the original into a "processed" subfolder so nothing runs twice

--------------------------------------------------------------------
SETUP (one time):
    pip install rembg[cpu] pillow onnxruntime

  First run downloads the AI model (~170 MB) to a local cache,
  after that everything runs offline.

RUN:
    python boot_photo_processor.py
--------------------------------------------------------------------
"""

import os
import sys
import time
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from rembg import remove, new_session

# =====================================================================
# CONFIG  (edit these, nothing below should need it)
# =====================================================================

# Folder where your phone photos land (Google Drive for Desktop sync path).
WATCH_FOLDER = r"G:\My Drive\Background remover"

# Where finished, edited photos are saved.
OUTPUT_FOLDER = r"C:\Users\barri\OneDrive\Pictures\Sideline Swap Photos"

# Originals get moved here after processing (inside WATCH_FOLDER).
PROCESSED_SUBFOLDER = "processed_originals"

# --- Backdrop ---------------------------------------------------------
BACKGROUND_STYLE = "studio_black"   # "white" or "studio_black"

CANVAS_SIZE = (1600, 1600)   # final image size
PADDING_PCT = 0.08           # side/top margin around the boot
JPEG_QUALITY = 92
OUTPUT_FORMAT = "JPEG"       # "JPEG" or "PNG" (PNG + "white" = transparent bg)

# White style
WHITE_BG_COLOR = (255, 255, 255)

# Studio black style
FLOOR_LINE_PCT = 0.60            # where the wall/floor fold sits (0.70 = 70% down)
WALL_TOP_COLOR = (10, 10, 12)    # near-black at top of wall
WALL_BOTTOM_COLOR = (52, 52, 58) # wall brightens toward the fold
FLOOR_NEAR_COLOR = (30, 30, 34)  # floor at the fold
FLOOR_EDGE_COLOR = (14, 14, 16)  # floor darkens toward the bottom edge
FOLD_SOFTNESS = 20               # blur radius of the fold (bigger = softer curve)
SHADOW_OPACITY = 120             # 0-255. 0 disables the drop shadow
SHADOW_BLUR = 27                 # softness of the shadow edge

# --- Color correction (all mild; set to False/1.0 to disable) --------
AUTO_WHITE_BALANCE = True
CONTRAST_BOOST = 1.08
SATURATION_BOOST = 1.06
BRIGHTNESS_BOOST = 1.02

# --- Background removal ----------------------------------------------
REMBG_MODEL = "birefnet-general"
ALPHA_MATTING = False        # True = cleaner fuzzy edges, slower

# --- Watcher behavior -------------------------------------------------
POLL_SECONDS = 5
STABLE_CHECKS = 2            # file size unchanged this many polls before processing
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}

# =====================================================================
# END CONFIG
# =====================================================================


def auto_white_balance(img: Image.Image) -> Image.Image:
    """Simple gray-world white balance, clamped so it stays subtle."""
    r, g, b = img.split()
    r_avg = ImageOps.grayscale(r).resize((1, 1)).getpixel((0, 0))
    g_avg = ImageOps.grayscale(g).resize((1, 1)).getpixel((0, 0))
    b_avg = ImageOps.grayscale(b).resize((1, 1)).getpixel((0, 0))
    avg = (r_avg + g_avg + b_avg) / 3.0

    def scale(channel, ch_avg):
        if ch_avg == 0:
            return channel
        factor = max(0.85, min(1.15, avg / ch_avg))
        return channel.point(lambda p: min(255, int(p * factor)))

    return Image.merge("RGB", (scale(r, r_avg), scale(g, g_avg), scale(b, b_avg)))


def color_correct(img: Image.Image) -> Image.Image:
    if AUTO_WHITE_BALANCE:
        img = auto_white_balance(img)
    if CONTRAST_BOOST != 1.0:
        img = ImageEnhance.Contrast(img).enhance(CONTRAST_BOOST)
    if SATURATION_BOOST != 1.0:
        img = ImageEnhance.Color(img).enhance(SATURATION_BOOST)
    if BRIGHTNESS_BOOST != 1.0:
        img = ImageEnhance.Brightness(img).enhance(BRIGHTNESS_BOOST)
    return img


def make_studio_backdrop(size) -> Image.Image:
    """Dark wall + floor with a soft cove 'fold' where they meet."""
    w, h = size
    floor_y = int(h * FLOOR_LINE_PCT)
    bg = Image.new("RGB", size)
    px = bg.load()

    for y in range(h):
        if y < floor_y:
            t = y / max(1, floor_y)                      # 0 at top -> 1 at fold
            c = tuple(
                int(WALL_TOP_COLOR[i] + (WALL_BOTTOM_COLOR[i] - WALL_TOP_COLOR[i]) * t)
                for i in range(3)
            )
        else:
            t = (y - floor_y) / max(1, h - floor_y)      # 0 at fold -> 1 at bottom
            c = tuple(
                int(FLOOR_NEAR_COLOR[i] + (FLOOR_EDGE_COLOR[i] - FLOOR_NEAR_COLOR[i]) * t)
                for i in range(3)
            )
        for x in range(w):
            px[x, y] = c

    # Soften the wall/floor transition into a curved cove fold
    if FOLD_SOFTNESS > 0:
        band_top = max(0, floor_y - FOLD_SOFTNESS * 2)
        band_bot = min(h, floor_y + FOLD_SOFTNESS * 2)
        band = bg.crop((0, band_top, w, band_bot)).filter(
            ImageFilter.GaussianBlur(FOLD_SOFTNESS)
        )
        bg.paste(band, (0, band_top))

    # Subtle center vignette-in-reverse: brighten where the boot will sit
    glow = Image.new("L", size, 0)
    gd = ImageDraw.Draw(glow)
    gd.ellipse(
        (w * 0.18, floor_y - h * 0.28, w * 0.82, floor_y + h * 0.18), fill=40
    )
    glow = glow.filter(ImageFilter.GaussianBlur(120))
    lighter = ImageEnhance.Brightness(bg).enhance(1.6)
    bg = Image.composite(lighter, bg, glow)
    return bg


def composite_on_backdrop(cutout: Image.Image) -> Image.Image:
    """Place the cutout on the configured backdrop and return final RGB(A)."""
    canvas_w, canvas_h = CANVAS_SIZE
    pad = int(min(canvas_w, canvas_h) * PADDING_PCT)

    if BACKGROUND_STYLE == "studio_black":
        floor_y = int(canvas_h * FLOOR_LINE_PCT)
        # Boot must fit between top padding and a little past the fold
        max_w = canvas_w - 2 * pad
        max_h = int(floor_y + canvas_h * 0.30) - pad
        cutout.thumbnail((max_w, max_h), Image.LANCZOS)

        bg = make_studio_backdrop(CANVAS_SIZE)
        x = (canvas_w - cutout.width) // 2
        # Bottom of boot sits just below the fold -> "standing on the floor"
        y = int(floor_y + canvas_h * 0.29) - cutout.height

        # Drop shadow: soft ellipse under the boot
        if SHADOW_OPACITY > 0:
            shadow = Image.new("L", CANVAS_SIZE, 0)
            sd = ImageDraw.Draw(shadow)
            sh_w = cutout.width * 1.05
            sh_h = max(20, int(cutout.width * 0.16))
            cx = x + cutout.width / 2
            cy = y + cutout.height - sh_h * 0.59
            sd.ellipse(
                (cx - sh_w / 2, cy - sh_h / 2, cx + sh_w / 2, cy + sh_h / 2),
                fill=SHADOW_OPACITY,
            )
            shadow = shadow.filter(ImageFilter.GaussianBlur(SHADOW_BLUR))
            black = Image.new("RGB", CANVAS_SIZE, (0, 0, 0))
            bg = Image.composite(black, bg, shadow)

        bg.paste(cutout, (x, y), mask=cutout.split()[3])
        return bg

    # --- white style (original behavior) ---
    max_w, max_h = canvas_w - 2 * pad, canvas_h - 2 * pad
    cutout.thumbnail((max_w, max_h), Image.LANCZOS)
    if OUTPUT_FORMAT.upper() == "PNG":
        canvas = Image.new("RGBA", CANVAS_SIZE, (0, 0, 0, 0))
    else:
        canvas = Image.new("RGB", CANVAS_SIZE, WHITE_BG_COLOR)
    x = (canvas_w - cutout.width) // 2
    y = (canvas_h - cutout.height) // 2
    canvas.paste(cutout, (x, y), mask=cutout.split()[3])
    return canvas


def process_image(src_path: Path, session) -> Path:
    # Open inside a context manager and force a full load so the file
    # handle is released immediately (Windows locks open files).
    with Image.open(src_path) as im:
        im.load()
        img = ImageOps.exif_transpose(im).convert("RGB")

    img = color_correct(img)

    cutout = remove(img, session=session, alpha_matting=ALPHA_MATTING)

    bbox = cutout.getbbox()
    if bbox:
        cutout = cutout.crop(bbox)

    canvas = composite_on_backdrop(cutout)

    out_dir = Path(OUTPUT_FOLDER)
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = ".png" if OUTPUT_FORMAT.upper() == "PNG" else ".jpg"
    out_path = out_dir / (src_path.stem + "_edited" + ext)

    if OUTPUT_FORMAT.upper() == "PNG":
        canvas.save(out_path, "PNG")
    else:
        canvas.convert("RGB").save(out_path, "JPEG", quality=JPEG_QUALITY, optimize=True)

    return out_path


def safe_move(src: Path, dst: Path, attempts: int = 6, delay: float = 2.0) -> bool:
    """Move a file, retrying if Windows/Drive still holds a lock on it.
    Falls back to copy if it never unlocks. Returns True if src is gone."""
    for i in range(attempts):
        try:
            shutil.move(str(src), str(dst))
            return True
        except PermissionError:
            time.sleep(delay)
    # Still locked after all retries: copy it so the original stays put,
    # and let the caller mark it as handled so it isn't reprocessed.
    try:
        shutil.copy2(str(src), str(dst))
        print("(locked, copied instead) ", end="")
    except Exception:
        pass
    return False


def wait_for_stable_files(folder: Path, seen_sizes: dict) -> list:
    ready = []
    for f in folder.iterdir():
        if not f.is_file() or f.suffix.lower() not in VALID_EXTENSIONS:
            continue
        size = f.stat().st_size
        history = seen_sizes.setdefault(f.name, [])
        history.append(size)
        if len(history) > STABLE_CHECKS:
            history.pop(0)
        if len(history) == STABLE_CHECKS and len(set(history)) == 1 and size > 0:
            ready.append(f)
    return ready


def main():
    watch = Path(WATCH_FOLDER)
    if not watch.exists():
        print(f"[!] Watch folder not found: {watch}")
        print("    Edit WATCH_FOLDER at the top of this script.")
        sys.exit(1)

    processed_dir = watch / PROCESSED_SUBFOLDER
    processed_dir.mkdir(exist_ok=True)

    print("Loading background-removal model (first run downloads ~170 MB)...")
    session = new_session(REMBG_MODEL)
    print(f"Watching: {watch}")
    print(f"Output:   {OUTPUT_FOLDER}")
    print(f"Backdrop: {BACKGROUND_STYLE}")
    print("Drop photos in the watch folder. Ctrl+C to stop.\n")

    seen_sizes = {}
    already_done = set()  # files processed but left in place (lock fallback)

    while True:
        try:
            for f in wait_for_stable_files(watch, seen_sizes):
                if f.name in already_done:
                    continue
                print(f"-> {f.name} ... ", end="", flush=True)
                try:
                    out = process_image(f, session)
                    moved = safe_move(f, processed_dir / f.name)
                    if moved:
                        seen_sizes.pop(f.name, None)
                    else:
                        already_done.add(f.name)
                    print(f"done -> {out.name}")
                except Exception as e:
                    print(f"FAILED ({e})")
                    seen_sizes[f.name] = ["FAILED"]
            time.sleep(POLL_SECONDS)
        except KeyboardInterrupt:
            print("\nStopped.")
            break


if __name__ == "__main__":
    main()
