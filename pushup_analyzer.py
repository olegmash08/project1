"""
Push-Up Quality Analyzer
========================
Uses webcam + MediaPipe Pose to count reps and evaluate push-up form.
Press 'q' to quit.
"""

import cv2
import mediapipe as mp
import numpy as np
import sys
import time

# ─── MediaPipe setup ────────────────────────────────────────────────────────
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles


# ─── Helper functions ───────────────────────────────────────────────────────

def calc_angle(a, b, c):
    """
    Returns the angle (in degrees) at point B formed by A-B-C.
    a, b, c are (x, y) tuples / arrays.
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def get_point(landmarks, idx, w, h):
    """Return pixel (x, y) for a given landmark index."""
    lm = landmarks[idx]
    return int(lm.x * w), int(lm.y * h)


def get_xy(landmarks, idx):
    """Return (x, y) normalised for angle maths."""
    lm = landmarks[idx]
    return lm.x, lm.y


# ─── Analyser class ─────────────────────────────────────────────────────────

class PushUpAnalyzer:
    DOWN_ANGLE = 100   # elbow angle considered "down" position
    UP_ANGLE   = 150   # elbow angle considered "up" position
    MAX_HIP_DEVIATION = 0.08  # normalised units – how much hip can stray from body line

    def __init__(self):
        self.rep_count  = 0
        self.stage      = "UP"   # current phase
        self.feedbacks  = []     # list of form messages for this rep
        self._in_rep    = False  # True while we are in the DOWN phase

    # ── core analysis ──────────────────────────────────────────────────────

    def analyse(self, landmarks, w, h):
        """
        Process one frame's landmarks.
        Returns a dict with analysis results.
        """
        lm = landmarks

        # ── elbow angles (average left & right) ──────────────────────────
        # Left: 11-13-15  /  Right: 12-14-16
        l_elbow = calc_angle(get_xy(lm, 11), get_xy(lm, 13), get_xy(lm, 15))
        r_elbow = calc_angle(get_xy(lm, 12), get_xy(lm, 14), get_xy(lm, 16))
        elbow_angle = (l_elbow + r_elbow) / 2

        # ── body alignment (hip sag / pike) ──────────────────────────────
        # Shoulder mid-point → hip mid-point → ankle mid-point should be ~straight
        sh_y  = (lm[11].y + lm[12].y) / 2
        hip_y = (lm[23].y + lm[24].y) / 2
        an_y  = (lm[27].y + lm[28].y) / 2

        # In push-up top-down view y increases downward.
        # Interpolate where the hip SHOULD be on the shoulder→ankle line.
        sh_x  = (lm[11].x + lm[12].x) / 2
        an_x  = (lm[27].x + lm[28].x) / 2
        hip_x = (lm[23].x + lm[24].x) / 2

        # Approximate: project hip onto shoulder-ankle segment
        if abs(an_y - sh_y) > 0.01:
            t = (hip_y - sh_y) / (an_y - sh_y)
        else:
            t = 0.5
        expected_hip_x = sh_x + t * (an_x - sh_x)
        hip_deviation  = abs(hip_x - expected_hip_x)   # lateral deviation (if camera is side-on)

        # Simpler sag/pike: compare hip_y to the ideal line between shoulder and ankle
        ideal_hip_y = sh_y + t * (an_y - sh_y)        # already == hip_y if perfectly aligned
        vertical_dev = hip_y - ideal_hip_y             # + = hips high (pike), – = hips drop (sag)

        # ── phase detection & rep counting ───────────────────────────────
        stage_changed = False
        if self.stage == "UP" and elbow_angle < self.DOWN_ANGLE:
            self.stage = "DOWN"
            self.feedbacks = []          # fresh feedback for new rep
            stage_changed = True
        elif self.stage == "DOWN" and elbow_angle > self.UP_ANGLE:
            self.stage = "UP"
            self.rep_count += 1
            stage_changed = True

        # ── form evaluation ──────────────────────────────────────────────
        form_issues = []

        if self.stage == "DOWN":
            if elbow_angle > 110:
                form_issues.append("Опускайтесь нижче (кут лікті > 110°)")
            if elbow_angle < 50:
                form_issues.append("Занадто низько – можливе навантаження на плечі")

        # Body alignment (works best with side camera view)
        if vertical_dev > self.MAX_HIP_DEVIATION * 2:
            form_issues.append("Таз завищений (піке) – вирівняйте корпус")
        elif vertical_dev < -self.MAX_HIP_DEVIATION * 2:
            form_issues.append("Таз провисає (прогин) – напружте прес")

        # Elbow flare (elbows should be roughly 45° from torso, not 90°)
        l_shoulder  = get_xy(lm, 11)
        l_elbow_pt  = get_xy(lm, 13)
        l_hip       = get_xy(lm, 23)
        elbow_torso = calc_angle(l_elbow_pt, l_shoulder, l_hip)
        if elbow_torso > 75:
            form_issues.append("Лікті розставлені надто широко")

        form_ok = len(form_issues) == 0

        return {
            "elbow_angle"  : elbow_angle,
            "l_elbow"      : l_elbow,
            "r_elbow"      : r_elbow,
            "stage"        : self.stage,
            "rep_count"    : self.rep_count,
            "form_ok"      : form_ok,
            "form_issues"  : form_issues,
            "stage_changed": stage_changed,
            "vertical_dev" : vertical_dev,
        }


# ─── Drawing helpers ─────────────────────────────────────────────────────────

def draw_overlay(frame, result, landmarks, w, h):
    """Draw info overlay on the frame."""
    GREEN  = (0, 220, 0)
    RED    = (0, 50, 220)
    YELLOW = (0, 200, 220)
    WHITE  = (255, 255, 255)
    DARK   = (20, 20, 20)

    # Semi-transparent sidebar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (340, 200), DARK, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Rep count
    cv2.putText(frame, f"Повторення: {result['rep_count']}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, WHITE, 2)

    # Stage
    stage_color = GREEN if result["stage"] == "UP" else YELLOW
    cv2.putText(frame, f"Фаза: {result['stage']}",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, stage_color, 2)

    # Elbow angle
    cv2.putText(frame, f"Кут лікті: {result['elbow_angle']:.0f} deg",
                (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.75, WHITE, 2)

    # Form
    form_color = GREEN if result["form_ok"] else RED
    form_text  = "Техніка: OK" if result["form_ok"] else "Техніка: ПОМИЛКА"
    cv2.putText(frame, form_text,
                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.75, form_color, 2)

    # Elbow angle arc on joint
    for idx in [13, 14]:
        pt = get_point(landmarks, idx, w, h)
        cv2.circle(frame, pt, 8, YELLOW, -1)

    return frame


# ─── Terminal output ──────────────────────────────────────────────────────────

# Accumulate per-rep issues across the last 10 reps
_rep_issues_log: list = []   # list of (rep_no, [issues])


def record_rep(result):
    """Save this rep's issues for the upcoming 10-rep summary."""
    _rep_issues_log.append((result["rep_count"], list(result["form_issues"])))


def print_10rep_summary(reps_done):
    """
    Print a summary covering the last batch of 10 reps.
    Called when rep_count hits a multiple of 10.
    """
    batch_start = reps_done - 9
    batch_end   = reps_done

    # Collect issues recorded in this batch
    batch_log = [(rn, iss) for rn, iss in _rep_issues_log
                 if batch_start <= rn <= batch_end]
    _rep_issues_log.clear()

    bad_reps   = [(rn, iss) for rn, iss in batch_log if iss]
    good_count = len(batch_log) - len(bad_reps)

    print()
    print("=" * 60)
    print(f"  ПІДСУМОК: повторення {batch_start}–{batch_end}")
    print(f"  Чисті повторення:  {good_count} / 10")
    print(f"  Із зауваженнями:   {len(bad_reps)} / 10")
    if bad_reps:
        print()
        print("  Деталі:")
        for rn, iss in bad_reps:
            for issue in iss:
                print(f"    • Повторення #{rn}: {issue}")
    print("=" * 60)
    print()


# ─── Main loop ────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Push-Up Quality Analyzer")
    print("  Натисніть 'q' у вікні камери, щоб вийти")
    print("=" * 60)
    print()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ПОМИЛКА: Не вдалося відкрити камеру.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    analyzer   = PushUpAnalyzer()
    prev_reps  = 0
    prev_time  = time.time()

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    ) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("\nНе вдалося зчитати кадр з камери.")
                break

            h, w = frame.shape[:2]

            # ── Pose detection ──────────────────────────────────────────
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)
            rgb.flags.writeable = True
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # ── FPS ─────────────────────────────────────────────────────
            now = time.time()
            fps = 1.0 / (now - prev_time + 1e-6)
            prev_time = now

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                # Draw skeleton
                mp_draw.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=3),
                    mp_draw.DrawingSpec(color=(245, 66, 230),  thickness=2, circle_radius=2),
                )

                # Analyse
                result = analyzer.analyse(lm, w, h)

                # Draw overlay
                draw_overlay(frame, result, lm, w, h)

                # Record & print every 10 reps
                if result["stage_changed"] and result["stage"] == "UP" and result["rep_count"] > 0:
                    record_rep(result)
                    if result["rep_count"] % 10 == 0:
                        print_10rep_summary(result["rep_count"])

            else:
                cv2.putText(frame, "Поза не виявлена – станьте перед камерою",
                            (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                sys.stdout.write("\rПоза не виявлена...".ljust(80))
                sys.stdout.flush()

            cv2.imshow("Push-Up Analyzer", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("\n\n" + "=" * 60)
    print(f"  Сесія завершена. Всього повторень: {analyzer.rep_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
