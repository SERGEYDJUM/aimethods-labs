from typing import Tuple
from PIL import ImageTk, Image
import cv2
import base64
import io
import numpy as np


def local_img_to_b64(path: str) -> bytes:
    status, png = cv2.imencode(".png", cv2.imread(path))
    assert status == True
    return base64.b64encode(png).decode("utf-8")


def b64_to_cv(base64_image: str) -> cv2.Mat:
    pilim = Image.open(io.BytesIO(base64.b64decode(base64_image)))
    return cv2.cvtColor(np.array(pilim), cv2.COLOR_BGR2RGB)


def cv2_to_tk(image: cv2.Mat, height=1024) -> ImageTk.PhotoImage:
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    scale = image.height / height
    image = image.resize(size=(int(image.width / scale), height))
    return ImageTk.PhotoImage(image=image)


def retrieve_image(stream) -> cv2.Mat:
    image = np.asarray(bytearray(stream.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


def psnr(noisy: cv2.Mat, truth: cv2.Mat) -> float:
    mse = np.mean(
        np.square(np.subtract(noisy.astype(np.float32), truth.astype(np.float32)))
    )

    if np.isclose(mse, 0):
        return np.Inf

    return 20 * np.log10(255.0) - 10 * np.log10(mse)
