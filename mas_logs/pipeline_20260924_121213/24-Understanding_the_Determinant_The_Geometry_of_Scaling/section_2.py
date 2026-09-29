from manim import *
import os

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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Defining the Determinant", [
            "Determinant is the area scaling factor.",
            "Transformation maps a shape to a region.",
            "It measures the expansion or shrinkage."
        ])
        
        # Assets
        # Placeholder SVG for [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg]
        icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg") if os.path.exists("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg") else Dot(color=YELLOW)
        square = Square(side_length=1.5, color=YELLOW, fill_opacity=0.3)
        label = Text("Area", font_size=24, color=YELLOW)
        square_and_label = VGroup(square, label, icon).arrange(DOWN)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.place_in_area(square_and_label, 'B4', 'C5', scale_factor=0.9)
        self.play(Create(square), Write(label), FadeIn(icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(BLUE)
        parallelogram = Polygon(
            np.array([-1, -1, 0]), 
            np.array([1, -1, 0]), 
            np.array([2, 1, 0]), 
            np.array([0, 1, 0]), 
            color=BLUE, fill_opacity=0.3
        )
        self.place_at_grid(parallelogram, 'D4', scale_factor=0.8)
        self.play(Transform(square, parallelogram), FadeOut(label), FadeOut(icon))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(RED)
        scale_text = Text("x5", font_size=30, color=RED)
        self.place_at_grid(scale_text, 'E5', scale_factor=1.0)
        self.play(Write(scale_text))
        self.wait(2)
