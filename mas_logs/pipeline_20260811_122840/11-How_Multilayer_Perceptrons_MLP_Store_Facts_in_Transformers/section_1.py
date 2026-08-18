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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Transformers contain two primary components.",
            "Attention layers act as dynamic readers.",
            "MLPs function as static, long-term memory.",
            "Think of MLPs as a giant database.",
            "They store facts like 'Paris is France's capital'."
        ]
        self.setup_layout("The Library inside the Transformer", lecture_lines)
        
        # Mobjects
        attention_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/library.svg")
        mlp_block = Square(color="#E74C3C", fill_opacity=0.5)
        
        # Initialize
        self.place_at_grid(attention_icon, "B4", scale_factor=0.6)
        attention_icon.set_color("#3498DB")
        self.place_at_grid(mlp_block, "B5", scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(attention_icon), FadeIn(mlp_block))
        self.lecture[0].set_color("#FFFF00")

        # === Animation for Lecture Line 2 ===
        self.play(attention_icon.animate.set_stroke(width=4), run_time=1)
        pulse = Circle(radius=0.5, color="#3498DB").move_to(attention_icon.get_center())
        self.play(Create(pulse), FadeOut(pulse), run_time=1.5)
        self.lecture[1].set_color("#3498DB")

        # === Animation for Lecture Line 3 ===
        glow = SurroundingRectangle(mlp_block, color="#E74C3C", buff=0.1, stroke_width=2)
        self.play(Create(glow), run_time=1)
        self.lecture[2].set_color("#E74C3C")

        # === Animation for Lecture Line 4 ===
        shelf = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/shelf.svg")
        self.place_at_grid(shelf, "D5", scale_factor=0.6)
        self.play(Transform(mlp_block, shelf), FadeOut(glow))
        self.lecture[3].set_color("#E74C3C")

        # === Animation for Lecture Line 5 ===
        fact_text = Text("Paris -> France", font_size=18, color=WHITE)
        self.place_at_grid(fact_text, "E5", scale_factor=0.8)
        self.play(Write(fact_text))
        self.lecture[4].set_color("#FF69B4")
        self.wait(2)
