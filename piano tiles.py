import cv2
import numpy as np
import random
import time
import pygame

from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------- CONFIG ----------------
LANES = 6
TILE_HEIGHT = 300
TILE_SPEED = 10          # faster than before
MAX_TILES = 2            # piano tiles constraint

BPM = 120                # change this to match your song
SECONDS_PER_BEAT = 60 / BPM
# ---------------------------------------

# ---------- AUDIO ----------
pygame.mixer.init()
pygame.mixer.music.load("/Users/sinemunlu/emotional-piano-music-256262.mp3")

# ---------- CAMERA ----------
cap = cv2.VideoCapture(0)

# ---------- HAND TRACKING ----------
base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

# ---------- GAME STATE ----------
tiles = []
score = 0
game_started = False
game_over = False

beat_start_time = 0
next_beat_index = 0

def spawn_tile():
    lane = random.randint(0, LANES - 1)
    return {"lane": lane, "y": -TILE_HEIGHT}

# ---------- MAIN LOOP ----------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    lane_width = w // LANES

    mp_image = Image(
        image_format=ImageFormat.SRGB,
        data=frame
    )
    result = detector.detect(mp_image)

    index_x = index_y = None
    thumbs_up = False

    if result.hand_landmarks:
        lm = result.hand_landmarks[0]
        index_x = int(lm[8].x * w)
        index_y = int(lm[8].y * h)

        # thumbs up
        if lm[4].y < lm[3].y:
            thumbs_up = True

    # ---------- START GAME ----------
    if not game_started and thumbs_up:
        game_started = True
        game_over = False
        tiles.clear()
        score = 0
        next_beat_index = 0
        beat_start_time = time.time()
        pygame.mixer.music.play()

    # ---------- GAME LOGIC ----------
    if game_started and not game_over:
        elapsed = time.time() - beat_start_time
        expected_beat = int(elapsed / SECONDS_PER_BEAT)

        # Spawn tiles on beats (max 2 on screen)
        if expected_beat >= next_beat_index and len(tiles) < MAX_TILES:
            tiles.append(spawn_tile())
            next_beat_index += 1

        # Move tiles
        for tile in tiles:
            tile["y"] += TILE_SPEED

        # Collision + miss detection
        for tile in tiles[:]:
            tx = tile["lane"] * lane_width
            ty = tile["y"]

            # Hit
            if index_x is not None:
                if tx < index_x < tx + lane_width and ty < index_y < ty + TILE_HEIGHT:
                    tiles.remove(tile)
                    score += 1
                    continue

            # Miss
            if ty > h:
                game_over = True
                pygame.mixer.music.stop()

    # ---------- DRAW ----------
    for i in range(1, LANES):
        cv2.line(frame, (i * lane_width, 0), (i * lane_width, h), (60, 60, 60), 2)

    for tile in tiles:
        tx = tile["lane"] * lane_width
        ty = tile["y"]
        cv2.rectangle(
            frame,
            (tx + 5, ty),
            (tx + lane_width - 5, ty + TILE_HEIGHT),
            (0, 0, 0),
            -1
        )

    if index_x is not None:
        cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), -1)

    if not game_started:
        cv2.putText(frame, "THUMBS UP TO START", (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    elif game_over:
        cv2.putText(frame, f"GAME OVER  Score: {score}", (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(frame, "Thumbs up to restart", (40, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
        if thumbs_up:
            game_started = False
    else:
        cv2.putText(frame, f"Score: {score}", (40, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Hand Piano Tiles", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()
