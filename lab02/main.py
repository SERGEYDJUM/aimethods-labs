from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from .model_infer import Model


class App(Tk):
    """Application UI"""

    base_font = ("CaskaydiaCove NF", 12)

    def __init__(self) -> None:
        """Initializes tkinter UI for this application"""
        
        Tk.__init__(self)
        self.model: Model = None
        self.temperature = 1.0
        self.token_cnt = 256

        self.wm_resizable(False, False)
        self.title("Генерация текста")

        self.input = Text(height=8, font=self.base_font)
        self.input.grid(row=0, column=0)

        # Controls begin
        self.btn_area = ttk.Frame()
        self.btn_area.grid(row=1, column=0)

        self.load_btn = ttk.Button(
            self.btn_area, text="Загрузить файл", command=self.load_file
        )
        self.load_btn.grid(row=0, column=0)

        self.exec_btn = ttk.Button(
            self.btn_area, text="Предсказать продолжение", command=self.predict
        )
        self.exec_btn.grid(row=0, column=1)

        self.token_cnt_label = ttk.Label(self.btn_area, text="Температура: ")
        self.token_cnt_label.grid(row=0, column=2, padx=5)

        self.temp_slider = ttk.Scale(
            self.btn_area,
            from_=0.0,
            to=2.0,
            value=self.temperature,
            orient=HORIZONTAL,
        )

        self.temp_slider.grid(row=0, column=3, padx=5)

        self.token_cnt_inp = ttk.Entry(self.btn_area, width=7)
        self.token_cnt_inp.insert(0, str(self.token_cnt))
        self.token_cnt_inp.grid(row=0, column=4)

        self.token_cnt_label = ttk.Label(self.btn_area, text="токенов")
        self.token_cnt_label.grid(row=0, column=5)
        # Controls end

        self.output = Text(font=self.base_font)
        self.output.grid(row=2, column=0)

    def load_file(self) -> None:
        """Opens a dialogue to pick input text file"""
        
        path = filedialog.askopenfilename()

        with open(path, encoding="utf-8") as file:
            text: str = file.read().strip()

            # Clear input Text
            self.input.delete("1.0", "999.999")
            self.input.insert("1.0", text)

    def predict(self) -> None:
        """Runs model using prompt from input Text and writes result into output Text"""

        # Load model on demand to shorten UI startup
        if self.model is None:
            self.model = Model()

        # Update generation parameters
        self.token_cnt = int(self.token_cnt_inp.get())
        self.temperature = float(self.temp_slider.get())

        # Get prompt from input
        prompt = self.input.get("1.0", "999.999").strip()
        
        print(f"[{self.temperature:.2f} | {self.token_cnt}] >>> {prompt}")

        if not prompt:
            return

        response = self.model.execute_prompt(
            prompt, temperature=self.temperature, max_new_tokens=self.token_cnt
        )

        response = response.strip().removeprefix(prompt).strip()

        print(f"<<< {response}")

        # Clear and paste text into output
        self.output.delete("1.0", "999.999")
        self.output.insert("1.0", response)


if __name__ == "__main__":
    App().mainloop()
