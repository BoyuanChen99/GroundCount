import os
from PIL import Image, ImageTk, ImageGrab
import tkinter as tk
import tkinter.ttk as ttk


class ImageViewer(tk.Tk):
    def __init__(self, images_info, title="", size="3000x2600", benchmark="POPE", save_mode=False):
        super().__init__()
        self.title(title)
        self.geometry(size)
        self.images_info = images_info
        self.idx = 0
        self.benchmark = benchmark
        self.save_mode = save_mode                       # <-- store flag
        self.screens_dir = os.path.join(os.getcwd(), "gui_screenshots")
        if self.save_mode:
            os.makedirs(self.screens_dir, exist_ok=True) # <-- ensure folder

        benchmarks_bool = ["pope"]
        benchmarks_with_answer = ["pope"]
        self.display_correctness = (benchmark.lower() in benchmarks_bool)
        self.display_answer = (benchmark.lower() in benchmarks_with_answer)

        self.img_label = tk.Label(self)
        self.img_label.pack(pady=50)

        self.task_label = tk.Label(self, font=("Helvetica", 24))
        self.task_label.pack(pady=20)

        main_font_size = 32
        self.info_text = tk.Text(self, height=8, wrap=tk.WORD, font=("Helvetica", main_font_size), padx=40, pady=40)
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=80)
        self.info_text.tag_configure("bold", font=("Helvetica", main_font_size, "bold"))

        if self.display_correctness:
            self.correctness_label = tk.Label(self, font=("Helvetica", 48, "bold"))
            self.correctness_label.pack(pady=10)
            self.info_text.tag_configure("yes", foreground="dark green", font=("Helvetica", 32, "bold"))
            self.info_text.tag_configure("no", foreground="dark red", font=("Helvetica", 32, "bold"))

        nav_frame = tk.Frame(self)
        nav_frame.pack(pady=50)

        self.prev_button = tk.Button(nav_frame, text="Previous", command=self.show_prev, font=("Helvetica", 32), height=3, width=16)
        self.prev_button.pack(side=tk.LEFT, padx=10)

        self.next_button = tk.Button(nav_frame, text="Next", command=self.show_next, font=("Helvetica", 32), height=3, width=16)
        self.next_button.pack(side=tk.LEFT, padx=10)

        self.select_button = tk.Button(nav_frame, text="Select Image", command=self.open_selector, font=("Helvetica", 32), height=3, width=16)
        self.select_button.pack(side=tk.LEFT, padx=10)

        self.show_image()


    def save_current_gui(self):
        """Capture the full app window and save with the required filename format."""
        if not self.images_info:
            return

        info = self.images_info[self.idx]

        # Determine C/F using model_output vs answer (case-insensitive)
        pred = str(info.get('model_output', '')).strip().lower()
        ans = str(info.get('answer', '')).strip().lower()
        correctness_char = 'C' if pred == ans and ans != '' else 'F'

        # Extract base image name (without extension)
        image_file = os.path.basename(str(info.get('image_file', 'unknown')))
        image_name, _ = os.path.splitext(image_file)

        # Map label (answer) YES/NO -> 1/0; default to 0 when not binary
        if ans == 'yes':
            q_bit = 1
        elif ans == 'no':
            q_bit = 0
        else:
            q_bit = 0  # fallback if label isn't strictly yes/no

        filename = f"{image_name}_Q{q_bit}_{correctness_char}.png"
        out_path = os.path.join(self.screens_dir, filename)

        # Compute window bbox in screen coords
        self.update_idletasks()
        x = self.winfo_rootx()
        y = self.winfo_rooty()
        w = self.winfo_width()
        h = self.winfo_height()
        bbox = (x, y, x + w, y + h)

        try:
            snap = ImageGrab.grab(bbox=bbox)
            snap.save(out_path)
            # Optional: print to console for debugging
            print(f"[saved] {out_path}")
        except Exception as e:
            print(f"[screenshot error] {e}")

    def show_image(self):
        if self.idx >= len(self.images_info):
            self.img_label.config(image='', text='Done!')
            if self.display_correctness:
                self.correctness_label.config(text='')
            self.info_text.delete("1.0", tk.END)
            self.info_text.insert(tk.END, "No more images.")
            self.next_button.config(state=tk.DISABLED)
            return

        self.next_button.config(state=tk.NORMAL)
        self.prev_button.config(state=tk.NORMAL if self.idx > 0 else tk.DISABLED)

        info = self.images_info[self.idx]

        # --- keep original aspect ratio with a fixed height ---
        FIXED_HEIGHT = 1200  # choose your constant display height
        pil_img = info['image']  # this is a PIL.Image already per your code
        orig_w, orig_h = pil_img.size

        # Optional: avoid upscaling very small images (remove 'min' to always force 1300 height)
        target_h = FIXED_HEIGHT
        scale = target_h / float(orig_h)
        target_w = int(round(orig_w * scale))

        pil_img_resized = pil_img.resize((target_w, target_h), resample=Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(pil_img_resized)
        self.img_label.config(image=self.tk_img)
        self.img_label.image = self.tk_img

        pred = str(info['model_output']).strip().lower()
        ans = str(info['answer']).strip().lower()
        if self.display_correctness:
            if pred == ans:
                self.correctness_label.config(text="CORRECT", fg="green")
            else:
                self.correctness_label.config(text="WRONG", fg="red")

        self.info_text.delete("1.0", tk.END)
        answer_string = f"Label: {info['answer']}\n" if self.display_answer else ""

        # Prompt
        self.info_text.insert(tk.END, f"{info['prompt']}\n")

        # VLM Output
        self.info_text.insert(tk.END, "VLM Output: ")
        start_out = self.info_text.index(tk.INSERT)

        model_out_raw = str(info['model_output']).strip()
        model_out_norm = model_out_raw.lower()
        model_out_disp = model_out_raw.upper() if model_out_norm in ("yes", "no") else model_out_raw

        self.info_text.insert(tk.END, f"{model_out_disp}\n", "bold")
        end_out = self.info_text.index(tk.INSERT)

        # --- Show original_response if available ---
        if "original_response" in info and info["original_response"]:
            self.info_text.insert(tk.END, "Original Response: ")
            start_orig = self.info_text.index(tk.INSERT)

            orig_raw = str(info['original_response']).strip()
            self.info_text.insert(tk.END, f"{orig_raw}\n")
            end_orig = self.info_text.index(tk.INSERT)


        # Label (if shown)
        if self.display_answer:
            self.info_text.insert(tk.END, "Label: ")
            start_lab = self.info_text.index(tk.INSERT)

            ans_raw = str(info['answer']).strip()
            ans_norm = ans_raw.lower()
            ans_disp = ans_raw.upper() if ans_norm in ("yes", "no") else ans_raw

            self.info_text.insert(tk.END, f"{ans_disp}\n")
            end_lab = self.info_text.index(tk.INSERT)

        # The rest of the info
        self.info_text.insert(tk.END,
            f"\nImage File: {info['image_file']}\n"
            f"Question ID: {info['question_id']}\n"
            f"Model ID: {info['model_id']}\n"
            f"Task: {info.get('task', 'N/A')}"
        )

        # Apply color tags only to the exact spans if they are YES/NO
        if self.display_correctness:
            if model_out_norm in ("yes", "no"):
                self.info_text.tag_add(model_out_norm, start_out, f"{start_out}+{len(model_out_disp)}c")
            if self.display_answer and ans_norm in ("yes", "no"):
                self.info_text.tag_add(ans_norm, start_lab, f"{start_lab}+{len(ans_disp)}c")


    def show_next(self):
        # Save the current GUI page BEFORE moving on
        if self.save_mode:
            self.save_current_gui()

        if self.idx < len(self.images_info) - 1:
            self.idx += 1
            self.show_image()

    def show_prev(self):
        if self.idx > 0:
            self.idx -= 1
            self.show_image()

    ### For the third button at the bottom
    def open_selector(self):
        selector = tk.Toplevel(self)
        selector.title("Select Image")
        selector.geometry("800x1000")

        # Create a custom style for a wider scrollbar handle
        style = ttk.Style(selector)
        style.theme_use("default")
        style.configure("Vertical.TScrollbar", gripcount=0, width=30, height=200, troughcolor='lightgray', bordercolor='gray', arrowcolor='black')

        # Use ttk Scrollbar (with larger handle)
        # Create custom scrollbar style with larger dimensions
        style = ttk.Style(selector)
        style.theme_use("default")

        # Wider scrollbar and bigger handle area
        style.configure("Vertical.TScrollbar",
            gripcount=1,
            width=40,                # <-- Wider scrollbar
            troughcolor='lightgray',
            background='darkgray',
            bordercolor='gray',
            arrowcolor='black'
        )

        # Taller scrollbar "thumb" (the draggable handle)
        style.layout("Vertical.TScrollbar",
            [
                ("Vertical.Scrollbar.trough", {
                    "children": [
                        ("Vertical.Scrollbar.thumb", {
                            "expand": "1",
                            "sticky": "nswe"
                        })
                    ],
                    "sticky": "ns"
                })
            ]
        )

        # Use this styled scrollbar
        scrollbar = ttk.Scrollbar(selector, orient=tk.VERTICAL, style="Vertical.TScrollbar")
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Larger font for listbox entries
        listbox_font = ("Helvetica", 32)
        listbox = tk.Listbox(selector, font=listbox_font, yscrollcommand=scrollbar.set, width=200)
        for i, info in enumerate(self.images_info):
            clean_file = os.path.splitext(info['image_file'])[0]  # Strip ".jpg"
            item = f"QID: {info['question_id']} | File: {clean_file}"
            listbox.insert(tk.END, item)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)

        def on_select(event):
            selection = listbox.curselection()
            if selection:
                self.idx = selection[0]
                self.show_image()
                selector.destroy()

        listbox.bind("<<ListboxSelect>>", on_select)