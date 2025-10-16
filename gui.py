import os
from pathlib import Path
from typing import Dict, List, Optional
import tkinter as tk
from tkinter import filedialog, messagebox

from detect_pickleball_court import SUPPORTED_FORMATS, process_image

FORMAT_LABELS = {
    "coco": "COCO Keypoints (.coco.json)",
    "yolo": "YOLOv8 Keypoints (.txt)",
    "cvat": "CVAT Points (.cvat.xml)",
    "keypoints_json": "Keypoints JSON (.keypoints.json)",
}


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master, padx=12, pady=12)
        self.master = master
        self.pack(fill="both", expand=True)
        self.input_folder: Optional[Path] = None
        self.output_folder: Optional[Path] = None
        self.format_vars: Dict[str, tk.BooleanVar] = {}
        self.save_overlay_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Select input and output folders to begin.")
        self.create_widgets()

    def create_widgets(self):
        self.master.title("Pickleball Court Detection")

        folder_frame = tk.LabelFrame(self, text="Folders", padx=10, pady=10)
        folder_frame.pack(fill="x", expand=False, pady=(0, 10))

        input_button = tk.Button(
            folder_frame,
            text="Select Input Folder",
            command=self.select_input_folder,
            width=25,
        )
        input_button.grid(row=0, column=0, sticky="w")
        self.input_folder_label = tk.Label(folder_frame, text="No folder selected", anchor="w")
        self.input_folder_label.grid(row=0, column=1, padx=(10, 0), sticky="w")

        output_button = tk.Button(
            folder_frame,
            text="Select Output Folder",
            command=self.select_output_folder,
            width=25,
        )
        output_button.grid(row=1, column=0, pady=(8, 0), sticky="w")
        self.output_folder_label = tk.Label(folder_frame, text="No folder selected", anchor="w")
        self.output_folder_label.grid(row=1, column=1, padx=(10, 0), pady=(8, 0), sticky="w")

        folder_frame.columnconfigure(1, weight=1)

        formats_frame = tk.LabelFrame(self, text="Annotation Formats", padx=10, pady=10)
        formats_frame.pack(fill="x", expand=False, pady=(0, 10))

        for idx, fmt in enumerate(SUPPORTED_FORMATS):
            label = FORMAT_LABELS.get(fmt, fmt.upper())
            default_selected = fmt == "coco"
            var = tk.BooleanVar(value=default_selected)
            chk = tk.Checkbutton(formats_frame, text=label, variable=var)
            chk.grid(row=idx, column=0, sticky="w", pady=2)
            self.format_vars[fmt] = var

        overlay_chk = tk.Checkbutton(
            formats_frame,
            text="Save visualization overlay image",
            variable=self.save_overlay_var,
        )
        overlay_chk.grid(row=len(SUPPORTED_FORMATS), column=0, sticky="w", pady=(8, 0))

        action_frame = tk.Frame(self)
        action_frame.pack(fill="x", expand=False, pady=(0, 10))

        self.run_button = tk.Button(action_frame, text="Run", command=self.run_batch, width=15)
        self.run_button.pack(side="left")

        quit_button = tk.Button(action_frame, text="Quit", command=self.master.destroy)
        quit_button.pack(side="right")

        status_label = tk.Label(
            self,
            textvariable=self.status_var,
            anchor="w",
            justify="left",
            wraplength=420,
        )
        status_label.pack(fill="x", expand=False, pady=(0, 5))

        self.progress_list = tk.Listbox(self, height=8)
        self.progress_list.pack(fill="both", expand=True)

    def select_input_folder(self):
        selected = filedialog.askdirectory()
        if selected:
            self.input_folder = Path(selected)
            self.input_folder_label.config(text=str(self.input_folder))
            self.status_var.set("Input folder selected.")

    def select_output_folder(self):
        selected = filedialog.askdirectory()
        if selected:
            self.output_folder = Path(selected)
            self.output_folder_label.config(text=str(self.output_folder))
            self.status_var.set("Output folder selected.")

    def run_batch(self):
        if not self.input_folder or not self.output_folder:
            messagebox.showwarning("Missing folders", "Please select both input and output folders.")
            return

        selected_formats = [fmt for fmt, var in self.format_vars.items() if var.get()]
        if not selected_formats:
            messagebox.showwarning("No formats", "Please select at least one annotation format to export.")
            return

        image_files = sorted(
            [Path(self.input_folder) / name for name in os.listdir(self.input_folder)
             if name.lower().endswith((".png", ".jpg", ".jpeg"))]
        )

        if not image_files:
            messagebox.showinfo("No images", "No PNG or JPG images found in the selected input folder.")
            return

        self.run_button.config(state="disabled")
        self.progress_list.delete(0, tk.END)
        errors: List[str] = []

        for index, image_path in enumerate(image_files, start=1):
            status_message = f"Processing {image_path.name} ({index}/{len(image_files)})"
            self.status_var.set(status_message)
            self.progress_list.insert(tk.END, status_message)
            self.progress_list.see(tk.END)
            self.update_idletasks()

            try:
                process_image(
                    image_path=image_path,
                    output_dir=self.output_folder,
                    formats=selected_formats,
                    save_overlay=self.save_overlay_var.get(),
                )
                self.progress_list.insert(tk.END, "  ✔ Completed")
            except Exception as exc:  # noqa: BLE001
                error_message = f"  ✖ {image_path.name}: {exc}"
                errors.append(error_message)
                self.progress_list.insert(tk.END, error_message)

            self.progress_list.see(tk.END)
            self.update_idletasks()

        self.run_button.config(state="normal")

        if errors:
            self.status_var.set(f"Completed with {len(errors)} error(s).")
            messagebox.showerror(
                "Processing finished with errors",
                "\n".join(errors[:5]) + ("\n..." if len(errors) > 5 else ""),
            )
        else:
            self.status_var.set("Processing complete.")
            messagebox.showinfo("Processing complete", f"Processed {len(image_files)} image(s) successfully.")


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("520x480")
    app = Application(master=root)
    app.mainloop()
