import tkinter as tk
class MiniMap:
    def __init__(self, root, image_matrix):
        self.root = root
        self.matrix = image_matrix
        self.root.title("Image Display")
        self.cell_size = 100
        self.canvas = tk.Canvas(root, width=len(image_matrix[0]) * self.cell_size, height=len(image_matrix) * self.cell_size)
        self.canvas.pack()
        self.color_dict = {0: "white", 1: "green", 2: "red", 3: "blue", 4: "yellow", 5: "black"}

        self.display_image()
        self.root.after(100, lambda : self.update_image(self.matrix))
        
    def display_image(self,):
        self.canvas.delete("all")  # Clear the canvas
        cell_size = self.cell_size  # Size of each cell in pixels
        for i, row in enumerate(self.matrix):
            for j, color_code in enumerate(row):
                color = self.color_dict[color_code]
                self.canvas.create_rectangle(j * cell_size, i * cell_size, (j + 1) * cell_size, (i + 1) * cell_size, fill=color, outline="black")

    def update_image(self,matrix):
        self.matrix = matrix
        self.display_image()
        self.root.after(100, lambda : self.update_image(self.matrix))
