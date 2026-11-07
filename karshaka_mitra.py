import cv2
import mediapipe as mp
import numpy as np
import math
import time

from PIL import Image, ImageDraw, ImageFont

# -------------------- SETTINGS --------------------

WIDTH = 1100
HEIGHT = 720

BG = (35, 48, 32)
CARD = (65, 91, 58)
LIGHT = (225, 235, 205)
ACCENT = (130, 190, 110)
WHITE = (255, 255, 255)

FONT_PATH = "assets/fonts/NotoSansMalayalam.ttf"

# -------------------- MALAYALAM FONT --------------------

font_cache = {}
text_cache = {}


def get_font(size):
    if size not in font_cache:
        try:
            font = ImageFont.truetype(
                FONT_PATH,
                size,
                layout_engine=ImageFont.Layout.RAQM
            )
        except (ValueError, AttributeError):
            font = ImageFont.truetype(FONT_PATH, size)

        font_cache[size] = font

    return font_cache[size]


def write(canvas, text, position, size=28, color=WHITE):

    key = (text, size, color)

    if key not in text_cache:

        font = get_font(size)

        bbox = font.getbbox(text)

        text_width = max(1, bbox[2] - bbox[0])
        text_height = max(1, bbox[3] - bbox[1])

        image = Image.new(
            "RGBA",
            (text_width + 4, text_height + 4),
            (0, 0, 0, 0)
        )

        draw = ImageDraw.Draw(image)

        # OpenCV colours are BGR; Pillow uses RGB.
        pil_color = (color[2], color[1], color[0], 255)

        draw.text(
            (2 - bbox[0], 2 - bbox[1]),
            text,
            font=font,
            fill=pil_color
        )

        rendered = np.array(image)

        bgr = cv2.cvtColor(rendered[:, :, :3], cv2.COLOR_RGB2BGR)
        alpha = rendered[:, :, 3] / 255.0

        text_cache[key] = (bgr, alpha)

    bgr, alpha = text_cache[key]

    x, y = position

    h, w = alpha.shape
    canvas_h, canvas_w = canvas.shape[:2]

    # Keep text inside the canvas.
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(canvas_w, x + w)
    y2 = min(canvas_h, y + h)

    if x1 >= x2 or y1 >= y2:
        return

    crop_x1 = x1 - x
    crop_y1 = y1 - y
    crop_x2 = crop_x1 + (x2 - x1)
    crop_y2 = crop_y1 + (y2 - y1)

    roi = canvas[y1:y2, x1:x2]

    text_part = bgr[crop_y1:crop_y2, crop_x1:crop_x2]
    alpha_part = alpha[crop_y1:crop_y2, crop_x1:crop_x2]

    alpha_part = alpha_part[:, :, np.newaxis]

    canvas[y1:y2, x1:x2] = (
        roi * (1 - alpha_part) + text_part * alpha_part
    ).astype(np.uint8)


# -------------------- PROJECT INFORMATION --------------------

crops = {
    "തക്കാളി": [
        "വെള്ളം: മണ്ണിന്റെ മുകൾഭാഗം ഉണങ്ങുമ്പോൾ നനയ്ക്കുക.",
        "സൂര്യപ്രകാശം: ദിവസവും 6 മുതല്‍ 8 മണിക്കൂര്‍.",
        "മണ്ണ്: വെള്ളം കെട്ടിനിൽക്കാത്ത ഫലഭൂയിഷ്ഠമായ മണ്ണ്.",
        "വളം: കമ്പോസ്റ്റ് അല്ലെങ്കിൽ ജൈവവളം ഉപയോഗിക്കുക.",
        "പരിചരണം: ചെടിക്ക് താങ്ങ് നൽകുക."
    ],

    "മുളക്": [
        "വെള്ളം: മണ്ണ് ഉണങ്ങുമ്പോൾ ആവശ്യത്തിന് നനയ്ക്കുക.",
        "സൂര്യപ്രകാശം: ദിവസവും 6 മുതല്‍ 8 മണിക്കൂര്‍.",
        "മണ്ണ്: നല്ല നീർവാർച്ചയുള്ള മണ്ണ്.",
        "വളം: കമ്പോസ്റ്റ് ഉപയോഗിക്കാം.",
        "പരിചരണം: കീടബാധ ശ്രദ്ധിക്കുക."
    ],

    "ചീര": [
        "വെള്ളം: മണ്ണിൽ ആവശ്യത്തിന് ഈർപ്പം നിലനിർത്തുക.",
        "സൂര്യപ്രകാശം: ദിവസവും 4 മുതല്‍ 6 മണിക്കൂര്‍.",
        "മണ്ണ്: ജൈവവളമുള്ള മണ്ണ്.",
        "വളം: കമ്പോസ്റ്റ് ചേർക്കാം.",
        "പരിചരണം: ഇലകൾ ആവശ്യത്തിന് വിളവെടുക്കുക."
    ]
}


symptoms = {
    "ഇല മഞ്ഞയാകുന്നു": [
        "സാധ്യമായ കാരണങ്ങൾ:",
        "നൈട്രജൻ പോലുള്ള പോഷകങ്ങളുടെ കുറവ്.",
        "അമിതമായ നനവ്.",
        "",
        "പരിഹാരങ്ങൾ:",
        "മണ്ണിലെ ഈർപ്പം പരിശോധിക്കുക.",
        "ആവശ്യത്തിന് ജൈവവളം നൽകുക."
    ],

    "ചെടി വാടുന്നു": [
        "സാധ്യമായ കാരണങ്ങൾ:",
        "വെള്ളത്തിന്റെ കുറവ് അല്ലെങ്കിൽ അധികം.",
        "വേരുകൾക്ക് കേടുപാടുകൾ.",
        "",
        "പരിഹാരങ്ങൾ:",
        "മണ്ണിലെ ഈർപ്പം പരിശോധിക്കുക.",
        "വെള്ളം കെട്ടിനിൽക്കുന്നില്ലെന്ന് ഉറപ്പാക്കുക."
    ],

    "ഇലയുടെ അരികുകൾ ഉണങ്ങുന്നു": [
        "സാധ്യമായ കാരണങ്ങൾ:",
        "വെള്ളത്തിന്റെ കുറവ്.",
        "അമിതമായ ചൂട്.",
        "പോഷകങ്ങളുടെ അസന്തുലിതാവസ്ഥ.",
        "",
        "പരിഹാരങ്ങൾ:",
        "ആവശ്യത്തിന് വെള്ളം നൽകുക.",
        "കടുത്ത വെയിലിൽ നിന്ന് സംരക്ഷിക്കുക."
    ]
}


# -------------------- MEDIAPIPE SETUP --------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)


# -------------------- PINCH DETECTION --------------------

def is_pinching(hand, width, height):

    thumb_x = hand[4].x * width
    thumb_y = hand[4].y * height

    index_x = hand[8].x * width
    index_y = hand[8].y * height

    distance = math.hypot(
        thumb_x - index_x,
        thumb_y - index_y
    )

    palm_size = math.hypot(
        (hand[0].x - hand[9].x) * width,
        (hand[0].y - hand[9].y) * height
    )

    return distance < palm_size * 0.25


# -------------------- DRAWING FUNCTIONS --------------------

def draw_button(canvas, rect, text, active=False):

    x1, y1, x2, y2 = rect

    colour = ACCENT if active else CARD

    cv2.rectangle(
        canvas,
        (x1, y1),
        (x2, y2),
        colour,
        -1
    )

    cv2.rectangle(
        canvas,
        (x1, y1),
        (x2, y2),
        LIGHT,
        2
    )

    font = get_font(28)
    bbox = font.getbbox(text)

    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    text_x = x1 + ((x2 - x1) - text_width) // 2
    text_y = y1 + ((y2 - y1) - text_height) // 2

    write(
        canvas,
        text,
        (text_x, text_y),
        28,
        WHITE
    )


def inside(point, rect):

    x, y = point
    x1, y1, x2, y2 = rect

    return x1 <= x <= x2 and y1 <= y <= y2


# -------------------- APPLICATION STATE --------------------

screen = "home"

selected_item = ""
selected_type = ""

previous_pinch = False

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera could not be opened.")
    exit()


# -------------------- MAIN APPLICATION --------------------

try:

    with HandLandmarker.create_from_options(options) as landmarker:

        while True:

            success, frame = camera.read()

            if not success:
                print("Could not read camera.")
                break

            frame = cv2.flip(frame, 1)

            cam_h, cam_w = frame.shape[:2]

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp = int(time.monotonic() * 1000)

            results = landmarker.detect_for_video(
                mp_image,
                timestamp
            )

            canvas = np.zeros(
                (HEIGHT, WIDTH, 3),
                dtype=np.uint8
            )

            canvas[:] = BG

            cursor = None
            pinching = False

            # -------------------- HAND TRACKING --------------------

            if results.hand_landmarks:

                hand = results.hand_landmarks[0]

                cursor = (
                    int(hand[8].x * WIDTH),
                    int(hand[8].y * HEIGHT)
                )

                pinching = is_pinching(
                    hand,
                    cam_w,
                    cam_h
                )

            buttons = []

            # -------------------- HOME SCREEN --------------------

            if screen == "home":

                write(
                    canvas,
                    "കർഷക മിത്ര",
                    (390, 100),
                    55,
                    WHITE
                )

                write(
                    canvas,
                    "നിങ്ങളുടെ കൃഷി സഹായി",
                    (390, 180),
                    28,
                    LIGHT
                )

                buttons = [
                    ((100, 270, 510, 510), "crops"),
                    ((590, 270, 1000, 510), "symptoms")
                ]

                draw_button(
                    canvas,
                    buttons[0][0],
                    "വിള പരിപാലനം"
                )

                draw_button(
                    canvas,
                    buttons[1][0],
                    "കൃഷി അറിവ്"
                )

            # -------------------- CROP SCREEN --------------------

            elif screen == "crops":

                write(
                    canvas,
                    "വിള പരിപാലനം",
                    (390, 110),
                    42,
                    WHITE
                )

                home_rect = (30, 30, 170, 90)

                buttons.append((home_rect, "home"))

                draw_button(
                    canvas,
                    home_rect,
                    "ഹോം"
                )

                for i, crop in enumerate(crops.keys()):

                    y = 210 + i * 135

                    rect = (250, y, 850, y + 100)

                    buttons.append((rect, crop))

                    draw_button(
                        canvas,
                        rect,
                        crop
                    )

            # -------------------- SYMPTOM SCREEN --------------------

            elif screen == "symptoms":

                write(
                    canvas,
                    "കൃഷി അറിവ്",
                    (390, 110),
                    42,
                    WHITE
                )

                home_rect = (30, 30, 170, 90)

                buttons.append((home_rect, "home"))

                draw_button(
                    canvas,
                    home_rect,
                    "ഹോം"
                )

                for i, symptom in enumerate(symptoms.keys()):

                    y = 210 + i * 135

                    rect = (250, y, 850, y + 100)

                    buttons.append((rect, symptom))

                    draw_button(
                        canvas,
                        rect,
                        symptom
                    )

            # -------------------- DETAIL SCREEN --------------------

            elif screen == "detail":

                back_rect = (30, 30, 170, 90)

                buttons.append((back_rect, "back"))

                draw_button(
                    canvas,
                    back_rect,
                    "തിരികെ"
                )

                write(
                    canvas,
                    selected_item,
                    (250, 100),
                    40,
                    WHITE
                )

                if selected_type == "crops":
                    information = crops[selected_item]

                else:
                    information = symptoms[selected_item]

                y = 210

                for line in information:

                    if line == "":
                        y += 20
                        continue

                    write(
                        canvas,
                        line,
                        (100, y),
                        25,
                        LIGHT
                    )

                    y += 65

            # -------------------- HOVER DETECTION --------------------

            hovered = None

            if cursor is not None:

                for rect, action in buttons:

                    if inside(cursor, rect):

                        hovered = action

                        x1, y1, x2, y2 = rect

                        cv2.rectangle(
                            canvas,
                            (x1, y1),
                            (x2, y2),
                            WHITE,
                            4
                        )

                        break

                cv2.circle(
                    canvas,
                    cursor,
                    12,
                    (0, 0, 255),
                    -1
                )

            # -------------------- PINCH SELECTION --------------------

            if pinching and not previous_pinch and hovered is not None:

                if screen == "home":

                    if hovered == "crops":
                        screen = "crops"

                    elif hovered == "symptoms":
                        screen = "symptoms"

                elif screen in ["crops", "symptoms"]:

                    if hovered == "home":
                        screen = "home"

                    else:
                        selected_item = hovered
                        selected_type = screen
                        screen = "detail"

                elif screen == "detail":

                    if hovered == "back":
                        screen = selected_type

            previous_pinch = pinching

            # -------------------- FOOTER --------------------

            write(
                canvas,
                "വിരൽ ചൂണ്ടി നീക്കുക  |  വിരലുകൾ ചേർത്ത് തിരഞ്ഞെടുക്കുക",
                (180, 665),
                22,
                LIGHT
            )

            cv2.imshow(
                "KARSHaka MITRA",
                canvas
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

finally:

    camera.release()
    cv2.destroyAllWindows()

    print("Karshaka Mitra closed.")