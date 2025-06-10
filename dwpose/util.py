# type: ignore
import math
import os

import cv2
import numpy as np

eps = 0.01


def load_state_dict(path):
    """
    Loads a state dictionary from a file.
    """
    _, ext = os.path.splitext(path)
    if ext == ".safetensors":
        import safetensors.torch

        state_dict = safetensors.torch.load_file(path, device="cpu")
    else:
        import torch

        state_dict = torch.load(path, map_location="cpu", weights_only=True)

    return state_dict


def hwc3(image):
    """
    Ensures an image is in HWC format with 3 channels.
    """
    assert image.dtype == np.uint8
    if image.ndim == 2:
        image = image[:, :, None]
    assert image.ndim == 3
    h, w, c = image.shape
    assert c in [1, 3, 4]
    if c == 3:
        return image
    elif c == 1:
        return np.concatenate([image] * 3, axis=2)
    else:
        color = image[:, :, 0:3].astype(np.float32)
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        combined = color * alpha + 255 * (1 - alpha)
        combined = combined.clip(0, 255).astype(np.uint8)
        return combined  # type: ignore[no-any-return]


def safe_resize(image, resolution, nearest=64):
    """
    Resizes an image to the specified resolution, padding if necessary.
    """
    h, w, c = image.shape
    k = float(resolution) / min(h, w)
    h = float(h) * k  # type: ignore[assignment]
    w = float(w) * k  # type: ignore[assignment]
    h = int(np.round(h / float(nearest)) * nearest)
    w = int(np.round(w / float(nearest)) * nearest)
    image = cv2.resize(
        image, (w, h), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA
    )
    return image


def draw_bodypose(canvas, candidate, subset):
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    stickwidth = 4

    limbSeq = [
        [2, 3],
        [2, 6],
        [3, 4],
        [4, 5],
        [6, 7],
        [7, 8],
        [2, 9],
        [9, 10],
        [10, 11],
        [2, 12],
        [12, 13],
        [13, 14],
        [2, 1],
        [1, 15],
        [15, 17],
        [1, 16],
        [16, 18],
        [3, 17],
        [6, 18],
    ]

    colors = [
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 255, 0],
        [170, 255, 0],
        [85, 255, 0],
        [0, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 255, 255],
        [0, 170, 255],
        [0, 85, 255],
        [0, 0, 255],
        [85, 0, 255],
        [170, 0, 255],
        [255, 0, 255],
        [255, 0, 170],
        [255, 0, 85],
    ]

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly(
                (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1
            )
            cv2.fillConvexPoly(canvas, polygon, colors[i])

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            x = int(x * W)
            y = int(y * H)
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)

    return canvas


def draw_handpose(canvas, all_hand_peaks, draw_type="full-pose"):
    if draw_type in ["body-pose", "hand-pose", "hand-mask"]:
        return canvas

    import matplotlib

    H, W, C = canvas.shape

    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [0, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [0, 9],
        [9, 10],
        [10, 11],
        [11, 12],
        [0, 13],
        [13, 14],
        [14, 15],
        [15, 16],
        [0, 17],
        [17, 18],
        [18, 19],
        [19, 20],
    ]

    # (person_number*2, 21, 2)
    for i in range(len(all_hand_peaks)):
        peaks = all_hand_peaks[i]
        peaks = np.array(peaks)

        if draw_type in ["full-pose", "hand-pose"]:
            for ie, e in enumerate(edges):
                x1, y1 = peaks[e[0]]
                x2, y2 = peaks[e[1]]

                x1 = int(x1 * W)
                y1 = int(y1 * H)
                x2 = int(x2 * W)
                y2 = int(y2 * H)
                if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                    cv2.line(
                        canvas,
                        (x1, y1),
                        (x2, y2),
                        matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0])
                        * 255,
                        thickness=2,
                    )
        keypoints = []
        for _, keyponit in enumerate(peaks):
            x, y = keyponit

            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                if draw_type == "pose":
                    cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
                else:
                    keypoints.append([x, y])

        if draw_type in ["face-hand-mask", "hand-mask"] and keypoints:
            cv2.fillPoly(
                canvas, pts=[cv2.convexHull(np.array(keypoints))], color=(255, 255, 255)
            )
    return canvas


def draw_facepose(canvas, all_lmks, draw_type="full-pose"):
    if draw_type in ["body-pose", "hand-pose", "hand-mask"]:
        return canvas

    H, W, C = canvas.shape
    for lmks in all_lmks:
        keypoints = []
        lmks = np.array(lmks)
        for lmk in lmks:
            x, y = lmk
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                if draw_type in ["full-pose", "face-pose"]:
                    cv2.circle(canvas, (x, y), 3, (255, 255, 255), thickness=-1)
                else:
                    keypoints.append((x, y))
        if draw_type in ["face-hand-mask", "face-mask"] and keypoints:
            cv2.fillPoly(
                canvas, pts=[cv2.convexHull(np.array(keypoints))], color=(255, 255, 255)
            )
    return canvas
