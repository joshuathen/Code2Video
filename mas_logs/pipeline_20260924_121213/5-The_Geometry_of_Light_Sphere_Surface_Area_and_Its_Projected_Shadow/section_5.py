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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Summary and Quick Quiz", [
            "Sphere surface is 4 times shadow area.",
            "Shadow area is pi times r squared.",
            "Calculate surface area from shadow easily."
        ])
        
        # Load asset
        sphere_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg")
        self.place_at_grid(sphere_icon, "B5", scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFD700")
        summary_formula = MathTex(r"A_{surface} = 4 \times A_{shadow}", font_size=32)
        self.place_in_area(summary_formula, "B2", "B4", scale_factor=0.8)
        self.play(FadeIn(summary_formula), FadeIn(sphere_icon))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FFFF")
        shadow_formula = MathTex(r"A_{shadow} = \pi r^2", font_size=32)
        self.place_in_area(shadow_formula, "C2", "C4", scale_factor=0.8)
        self.play(FadeIn(shadow_formula))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF69B4")
        quiz_text = Text("Quick Quiz: Shadow Area = 10cm²", font_size=20)
        answer_text = Text("Surface Area = 40cm²", font_size=24, color="#00FF00")
        
        self.place_in_area(quiz_text, "D2", "D5", scale_factor=0.85)
        self.place_at_grid(answer_text, "E3", scale_factor=1.0)
        
        self.play(FadeIn(quiz_text))
        self.wait(1)
        self.play(FadeIn(answer_text))
        self.wait(2)
