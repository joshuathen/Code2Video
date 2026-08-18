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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Shannon entropy formula defines information uncertainty.",
            "Guesses create partitions of the remaining words.",
            "Entropy measures how evenly we split partitions.",
            "Balanced splits maximize information gain.",
            "Uneven partitions represent low entropy."
        ]
        self.setup_layout("Mathematical Foundation: Shannon Entropy", lecture_lines)
        
        # Asset path
        asset_bar = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/bar.svg"
        
        # === Animation for Lecture Line 1 ===
        # Show Shannon Entropy formula: H(X) = -sum(p log p)
        formula = MathTex(r"H(X) = -\sum p(i) \log_2 p(i)", color="#00FF00")
        self.place_at_grid(formula, 'C2', scale_factor=1.0)
        self.play(Write(formula))
        self.lecture[0].set_color("#00FF00")

        # === Animation for Lecture Line 2 ===
        # Explain variables p (probability) and H (entropy)
        label_h = Text("H = Uncertainty", font_size=20, color="#1E90FF").next_to(formula, DOWN)
        label_p = Text("p = Probability", font_size=20, color="#1E90FF").next_to(label_h, DOWN)
        self.play(Write(label_h), Write(label_p))
        self.lecture[1].set_color("#1E90FF")

        # === Animation for Lecture Line 3 ===
        # Animate bar chart using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/bar.svg]
        bar_chart = SVGMobject(asset_bar, color="#ADFF2F")
        self.place_in_area(bar_chart, 'D3', 'F5', scale_factor=0.6)
        self.play(FadeIn(bar_chart))
        self.lecture[2].set_color("#ADFF2F")

        # === Animation for Lecture Line 4 ===
        # Update H value (simulation) as bars change shape
        h_val = Text("H = 4.2", font_size=24, color="#FFD700").next_to(bar_chart, UP)
        self.play(Write(h_val))
        self.lecture[3].set_color("#FFD700")

        # === Animation for Lecture Line 5 ===
        # Highlight the peak of the chart to show max entropy
        self.play(bar_chart.animate.set_color("#FF4500"))
        self.lecture[4].set_color("#FF4500")
        self.wait(2)
