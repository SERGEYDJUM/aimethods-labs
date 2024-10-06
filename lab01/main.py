import os
from typing import Callable
import cv2
from tkinter import *
from tkinter import ttk
from utils import *
from api import api_l, api_r

class App:
    def __init__(self) -> None:
        self.window = Tk()
        self.window.wm_resizable(False, False)
        self.window.title("Бенчмарк Сервисов Апскейлинга")

        self.cur_image = 0
        self.latent_strength = 0.3
        self.img_paths = [
            "./test_set/" + item
            for item in os.listdir("./test_set")
            if item.endswith(".png")
        ]

        if not self.img_paths:
            raise RuntimeError("No PNG images found")

        self.image_cache = {}
        self.original_image = cv2.imread(self.img_paths[self.cur_image])
        self.image_l = cv2_to_tk(self.original_image)
        self.image_r = cv2_to_tk(self.original_image)

        self.img_label_l = ttk.Label(self.window, image=self.image_l)
        self.img_label_l.grid(row=0, column=0)

        self.img_label_r = ttk.Label(self.window, image=self.image_r)
        self.img_label_r.grid(row=0, column=2)

        self.psnr_results = [
            [0.0 for _ in range(len(self.img_paths))],
            [0.0 for _ in range(len(self.img_paths))],
        ]

        self.psnr_label_l = ttk.Label(self.window, text="--")
        self.psnr_label_l.grid(row=1, column=0)

        self.psnr_label_r = ttk.Label(self.window, text="--")
        self.psnr_label_r.grid(row=1, column=2)

        self.button_back = ttk.Button(
            self.window, text="Предыдущее", command=self.prev_item, state=DISABLED
        )
        self.button_back.grid(row=2, column=0)

        self.button_process = ttk.Button(
            self.window, text="Запуск", command=self.benchmark
        )
        self.button_process.grid(row=2, column=1)

        self.button_forward = ttk.Button(
            self.window,
            text="Следующее",
            command=self.next_item,
            state=DISABLED if len(self.img_paths) == 1 else NORMAL,
        )
        self.button_forward.grid(row=2, column=2)

        self.strength_slider = ttk.Scale(
            self.window,
            value=0.3,
            length=512,
            command=lambda s: self.set_strength(float(s)),
        )
        self.strength_slider.grid(row=3, column=2)

    def next_item(self, forward=True):
        self.cur_image += 1 if forward else -1
        self.original_image = cv2.imread(self.img_paths[self.cur_image])
        self.image_l = cv2_to_tk(self.original_image)
        self.image_r = cv2_to_tk(self.original_image)

        self.img_label_l.config(image=self.image_l, borderwidth=0)
        self.img_label_r.config(image=self.image_r, borderwidth=0)

        self.button_back.config(
            state=NORMAL if (self.cur_image - 1) >= 0 else DISABLED,
        )

        self.button_forward.config(
            state=(NORMAL if (self.cur_image + 1) < len(self.img_paths) else DISABLED),
        )

        if self.psnr_results[0][self.cur_image] != 0:
            self.psnr_label_l.config(
                text=f"PSNR: {self.psnr_results[0][self.cur_image]:.2f}"
            )
            self.psnr_label_r.config(
                text=f"PSNR: {self.psnr_results[1][self.cur_image]:.2f}"
            )
        else:
            self.psnr_label_l.config(text="--")
            self.psnr_label_r.config(text="--")

    def prev_item(self):
        self.next_item(forward=False)

    def run(self):
        self.window.mainloop()

    def set_strength(self, s: float):
        print(f"Latent upscale strentgth set to: {s:.2f}")
        self.latent_strength = s

    def benchmark_side(self, api: Callable) -> cv2.Mat:
        api_idx = 0 if api == api_l else 1
        src_path = self.img_paths[self.cur_image]
        up_path = src_path.replace("test_set", f"upscaled_{api_idx+1}")
        down_path = src_path.replace("test_set", "downscaled")

        downscaled = cv2.resize(
            cv2.imread(src_path), dsize=(512, 512), interpolation=cv2.INTER_LANCZOS4
        )

        cv2.imwrite(down_path, downscaled)

        upscaled = (
            api(down_path) if api_idx == 0 else api(down_path, self.latent_strength)
        )

        cv2.imwrite(up_path, upscaled)

        self.psnr_results[api_idx][self.cur_image] = psnr(upscaled, self.original_image)

        return cv2_to_tk(upscaled)

    def benchmark(self):
        os.makedirs("./downscaled", exist_ok=True)
        os.makedirs("./upscaled_1", exist_ok=True)
        os.makedirs("./upscaled_2", exist_ok=True)

        print("Upscaling left...", flush=True)
        if self.psnr_results[0][self.cur_image] == 0.0:
            self.image_l = self.benchmark_side(api_l)
            self.img_label_l.config(image=self.image_l, borderwidth=3)

        self.psnr_label_l.config(
            text=f"PSNR: {self.psnr_results[0][self.cur_image]:.2f}"
        )

        print("Upscaling right...", flush=True)
        if self.psnr_results[1][self.cur_image] == 0.0:
            self.image_r = self.benchmark_side(api_r)
            self.img_label_r.config(image=self.image_r, borderwidth=3)

        self.psnr_label_r.config(
            text=f"PSNR: {self.psnr_results[1][self.cur_image]:.2f}"
        )
        print("Done")


if __name__ == "__main__":
    App().run()
