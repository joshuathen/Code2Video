from manim import *
import numpy as np

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Emergence of Newton Fractals", 
                          ["Assign colors to specific roots.", 
                           "Calculate convergence for every pixel.", 
                           "Fractals emerge at basin boundaries."])
        
        # Use assets as proxies for pixels/basins
        pixel_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/pixel.svg"
        basin1 = SVGMobject(pixel_path)
        basin2 = SVGMobject(pixel_path)
        basin3 = SVGMobject(pixel_path)

        # Apply positioning constraints from issues 29, 30, 31
        self.place_at_grid(basin1, 'A2', scale_factor=0.6)
        self.place_at_grid(basin2, 'A4', scale_factor=0.6)
        self.place_at_grid(basin3, 'A6', scale_factor=0.6)

        # Labeling (B018)
        label1 = Text("Root 1", font_size=16, color="#FF0000").next_to(basin1, DOWN, buff=0.1)
        label2 = Text("Root 2", font_size=16, color="#00FF00").next_to(basin2, DOWN, buff=0.1)
        label3 = Text("Root 3", font_size=16, color="#0000FF").next_to(basin3, DOWN, buff=0.1)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF0000"))
        self.play(FadeIn(basin1), Write(label1))
        self.play(basin1.animate.set_color("#FF0000"))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        self.play(FadeIn(basin2), Write(label2))
        self.play(basin2.animate.set_color("#00FF00"))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#0000FF"))
        self.play(FadeIn(basin3), Write(label3))
        self.play(basin3.animate.set_color("#0000FF"))
        
        self.wait(2)
