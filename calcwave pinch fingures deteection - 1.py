import cv2
import time
import math
import ast
import re

# =========================
# Safe arithmetic evaluator
# =========================
# Supports: numbers, + - * /, parentheses, unary +/-
# No variables, no function calls, no attributes, no subscripts.
ALLOWED_CHARS = re.compile(r'^[0-9+\-*/().\s]+$')

def _sanitize(expr: str) -> str:
    # Allow common unicode symbols and accidental 'x'
    return (expr.replace('×', '*')
                .replace('x', '*')
                .replace('X', '*')
                .replace('÷', '/'))

def _eval_ast(node):
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)

    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)

        if isinstance(node.op, ast.Add):
            return left + right
        elif isinstance(node.op, ast.Sub):
            return left - right
        elif isinstance(node.op, ast.Mult):
            return left * right
        elif isinstance(node.op, ast.Div):
            if right == 0:
                raise ZeroDivisionError("division by zero")
            return left / right
        else:
            raise ValueError("Operator not allowed")

    if isinstance(node, ast.UnaryOp):
        val = _eval_ast(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +val
        elif isinstance(node.op, ast.USub):
            return -val
        else:
            raise ValueError("Unary operator not allowed")

    # Py 3.8+
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("Only numbers allowed")

    # Py <3.8
    if hasattr(ast, "Num") and isinstance(node, ast.Num):
        return float(node.n)

    # Parentheses are represented implicitly in the AST, so nothing to do.
    raise ValueError("Disallowed expression")

def safe_eval(expr: str):
    expr = _sanitize(expr.strip())
    if not expr:
        return ""

    if not ALLOWED_CHARS.match(expr):
        return "Error"

    try:
        # Parse to AST and strictly evaluate
        node = ast.parse(expr, mode='eval')
        result = _eval_ast(node)

        # Clean result: show "8" instead of "8.0" if it's an integer
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return result
    except Exception:
        return "Error"


# =========================
# Try MediaPipe (optional)
# =========================
MP_AVAILABLE = False
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    MP_AVAILABLE = True
except Exception:
    # No hard exit: we gracefully fall back to mouse mode.
    MP_AVAILABLE = False


# =========================
# UI components
# =========================
class Button:
    def __init__(self, x, y, w, h, label):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.label = label

    def draw(self, img, hover=False):
        base = (255, 255, 255) if not hover else (220, 235, 255)
        border = (30, 30, 30)
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), border, -1)
        cv2.rectangle(img, (self.x + 2, self.y + 2), (self.x + self.w - 2, self.y + self.h - 2), base, -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(self.label, font, 1, 2)
        tx = self.x + (self.w - tw) // 2
        ty = self.y + (self.h + th) // 2
        cv2.putText(img, self.label, (tx, ty), font, 1, (0, 0, 0), 2)

    def hit(self, px, py):
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h


def build_buttons(origin=(30, 140), bw=90, bh=70, gap=10):
    layout = [
        ["(", ")", "C", "⌫"],
        ["7", "8", "9", "/"],
        ["4", "5", "6", "*"],
        ["1", "2", "3", "-"],
        ["0", ".", "=", "+"],
    ]
    buttons = []
    ox, oy = origin
    for r, row in enumerate(layout):
        for c, lab in enumerate(row):
            x = ox + c * (bw + gap)
            y = oy + r * (bh + gap)
            buttons.append(Button(x, y, bw, bh, lab))
    return buttons


# =========================
# Helpers
# =========================
def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# =========================
# Main app
# =========================
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow("Gesture Calculator")
    buttons = build_buttons()

    expression = ""
    result_text = ""
    click_cooldown = 0
    CLICK_COOLDOWN_FRAMES = 12

    # Pointer state
    smooth_px, smooth_py = 0, 0
    SMOOTHING = 0.4
    pinch_threshold = 40  # pixels

    # Mouse fallback state
    mouse_x, mouse_y = -1, -1

    def press_label(label: str):
        nonlocal expression, result_text, click_cooldown
        if label == "C":
            expression, result_text = "", ""
        elif label == "⌫":
            if expression:
                expression = expression[:-1]
            result_text = ""
        elif label == "=":
            result = safe_eval(expression)
            result_text = str(result)
        else:
            if len(expression) < 64:
                expression += label
            result_text = ""
        click_cooldown = CLICK_COOLDOWN_FRAMES

    # Mouse click support (works even if MediaPipe is present)
    def on_mouse(event, x, y, flags, param):
        nonlocal mouse_x, mouse_y
        mouse_x, mouse_y = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            for b in buttons:
                if b.hit(x, y):
                    press_label(b.label)
                    break

    cv2.setMouseCallback("Gesture Calculator", on_mouse)

    # MediaPipe context if available
    hands_ctx = None
    if MP_AVAILABLE:
        hands_ctx = mp_hands.Hands(
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Header & display area
            cv2.rectangle(frame, (0, 0), (w, 120), (245, 245, 245), -1)
            title = "Gesture Calculator (Pinch to click)" if MP_AVAILABLE else "Calculator (Mouse Clicks)"
            cv2.putText(frame, title, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 2)

            display_line = expression if not result_text else f"{expression} = {result_text}"
            cv2.rectangle(frame, (30, 70), (w - 30, 110), (255, 255, 255), -1)
            cv2.putText(frame, display_line[-60:], (40, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)

            # Gesture or mouse pointer
            index_tip = None
            thumb_tip = None

            if MP_AVAILABLE and hands_ctx is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = hands_ctx.process(rgb)
                if res.multi_hand_landmarks:
                    # Use the first detected hand
                    handLms = res.multi_hand_landmarks[0]
                    lm = handLms.landmark
                    ix, iy = int(lm[8].x * w), int(lm[8].y * h)   # Index tip
                    tx, ty = int(lm[4].x * w), int(lm[4].y * h)   # Thumb tip
                    index_tip, thumb_tip = (ix, iy), (tx, ty)

                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        frame, handLms, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(80, 80, 80), thickness=1, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(160, 160, 160), thickness=1)
                    )

            hover_idx = -1

            # Compute "pointer" for hover highlighting
            if index_tip is not None:
                px, py = index_tip
                smooth_px = int(SMOOTHING * px + (1 - SMOOTHING) * smooth_px)
                smooth_py = int(SMOOTHING * py + (1 - SMOOTHING) * smooth_py)

                for i, b in enumerate(buttons):
                    if b.hit(smooth_px, smooth_py):
                        hover_idx = i
                        break

                # Pinch == click
                if thumb_tip is not None and hover_idx != -1 and click_cooldown == 0:
                    if distance(index_tip, thumb_tip) < pinch_threshold:
                        press_label(buttons[hover_idx].label)
            else:
                # Mouse hover highlight
                if 0 <= mouse_x < w and 0 <= mouse_y < h:
                    for i, b in enumerate(buttons):
                        if b.hit(mouse_x, mouse_y):
                            hover_idx = i
                            break

            # Draw buttons
            for i, b in enumerate(buttons):
                b.draw(frame, hover=(i == hover_idx))

            # Draw gesture cursor
            if index_tip is not None:
                cv2.circle(frame, (smooth_px, smooth_py), 12, (50, 120, 250), -1)

            # Click cooldown
            if click_cooldown > 0:
                click_cooldown -= 1

            cv2.imshow("Gesture Calculator", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
    finally:
        if hands_ctx is not None:
            hands_ctx.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
