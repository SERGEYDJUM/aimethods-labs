import os
from tkinter import *
from tkinter import ttk
from typing import Tuple
from utils import *
import cv2
from api import api_l, api_r
import os


class App:
    def __init__(self) -> None:
        self.window = Tk()
        self.window.wm_resizable(False, False)
        self.window.title("Upscaling Benchmark")

        self.cur_image = 0
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

        self.button_back = ttk.Button(
            self.window, text="Back", command=self.prev_item, state=DISABLED
        )
        self.button_back.grid(row=1, column=0)

        self.button_process = ttk.Button(
            self.window, text="Process", command=self.benchmark
        )
        self.button_process.grid(row=1, column=1)

        self.button_forward = ttk.Button(
            self.window,
            text="Forward",
            command=self.next_item,
            state=DISABLED if len(self.img_paths) == 1 else NORMAL,
        )
        self.button_forward.grid(row=1, column=2)

        self.psnr_results = [
            ["--" for _ in range(len(self.img_paths))],
            ["--" for _ in range(len(self.img_paths))],
        ]

        self.psnr_label_l = ttk.Label(
            self.window, text=self.psnr_results[0][self.cur_image]
        )
        self.psnr_label_l.grid(row=2, column=0)

        self.psnr_label_r = ttk.Label(
            self.window, text=self.psnr_results[0][self.cur_image]
        )
        self.psnr_label_r.grid(row=2, column=2)

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

        self.psnr_label_l.config(text=self.psnr_results[0][self.cur_image])
        self.psnr_label_r.config(text=self.psnr_results[1][self.cur_image])

    def prev_item(self):
        self.next_item(forward=False)

    def run(self):
        self.window.mainloop()

    def benchmark(self):
        def benchmark_side(api_idx: int) -> Tuple[str, PhotoImage]:
            src_path = self.img_paths[self.cur_image]
            up_path = src_path.replace("test_set", f"upscaled_{api_idx+1}")
            down_path = src_path.replace("test_set", "downscaled")

            downscaled = cv2.resize(
                cv2.imread(src_path), dsize=(512, 512), interpolation=cv2.INTER_LANCZOS4
            )
            cv2.imwrite(down_path, downscaled)

            api = api_l if api_idx == 0 else api_r
            upscaled = api(down_path)
            cv2.imwrite(up_path, upscaled)

            return (f"{psnr(upscaled, self.original_image):.2f}", cv2_to_tk(upscaled))

        os.makedirs("./downscaled", exist_ok=True)
        os.makedirs("./upscaled_1", exist_ok=True)
        os.makedirs("./upscaled_2", exist_ok=True)

        if self.psnr_results[0][self.cur_image] == "--":
            self.psnr_results[0][self.cur_image], self.image_l = benchmark_side(0)
            self.img_label_l.config(image=self.image_l, borderwidth=3)

        self.psnr_label_l.config(text=self.psnr_results[0][self.cur_image])

        if self.psnr_results[1][self.cur_image] == "--":
            self.psnr_results[1][self.cur_image], self.image_r = benchmark_side(1)
            self.img_label_r.config(image=self.image_r, borderwidth=3)

        self.psnr_label_r.config(text=self.psnr_results[1][self.cur_image])


if __name__ == "__main__":
    App().run()
