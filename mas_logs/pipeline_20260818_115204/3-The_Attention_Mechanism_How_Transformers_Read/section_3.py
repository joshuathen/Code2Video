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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Attention is a matrix of similarity scores.",
            "Dot products turn queries and keys into scores.",
            "Softmax transforms scores into importance probabilities.",
            "High intensity shows strong attention relationships.",
            "Low intensity shows weak attention relationships."
        ]
        self.setup_layout("Visualizing the Score: The Softmax Heatmap", lecture_lines)
        
        # Create 5x5 grid
        grid_group = VGroup()
        for i in range(25):
            rect = Square(side_length=0.8, fill_opacity=0.3, color=WHITE, stroke_width=2)
            grid_group.add(rect)
        grid_group.arrange_in_grid(rows=5, cols=5, buff=0.1)
        # Apply fix from issue 32
        self.place_in_area(grid_group, 'A2', 'D5', scale_factor=0.75)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(grid_group))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(GRAY), self.lecture[1].animate.set_color("#FFD700"))
        # Apply fix from issue 34
        q_label = Text("Queries", font_size=20)
        k_label = Text("Keys", font_size=20)
        self.place_at_grid(k_label, 'A4', scale_factor=1.0)
        self.place_at_grid(q_label, 'C1', scale_factor=1.0)
        self.add(q_label, k_label)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(GRAY), self.lecture[2].animate.set_color("#00FFFF"))
        # Apply fix from issue 33
        formula = MathTex(r"\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum e^{x_j}}", font_size=24)
        self.place_at_grid(formula, 'E3', scale_factor=0.9)
        self.play(Write(formula))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[2].animate.set_color(GRAY), self.lecture[3].animate.set_color("#FF4500"))
        # Highlight a \"high\" intensity cell (e.g., 0,0)
        grid_group[0].set_fill("#FF4500", opacity=0.8)
        # Asset integration from issue 23
        try:
            animal_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/animal.svg")
            animal_icon.set_height(0.5)
            animal_icon.move_to(grid_group[0].get_center())
            self.play(FadeIn(animal_icon))
            self.play(grid_group[0].animate.scale(1.2), run_time=0.5)
            self.play(grid_group[0].animate.scale(1/1.2), run_time=0.5)
        except:
            pass

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[3].animate.set_color(GRAY), self.lecture[4].animate.set_color("#1E90FF"))
        # Highlight a \"low\" intensity cell (e.g., 4,4)
        grid_group[24].set_fill("#1E90FF", opacity=0.3)
        self.wait(2)
