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
        lecture_lines = ["Powers, roots, logs are linked.", "They are one numerical relationship.", "Each reveals a different secret."]
        self.setup_layout("The Unified Notation Map", lecture_lines)
        
        # Equations
        eq1 = MathTex("2^3=8", color=WHITE)
        eq2 = MathTex("8^{1/3}=2", color="#FFD700")
        eq3 = MathTex("\\log_2(8)=3", color="#FF4500")
        
        labels = [
            Text("Power", font_size=20, color="#32CD32"),
            Text("Root", font_size=20, color="#32CD32"),
            Text("Log", font_size=20, color="#32CD32")
        ]

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.place_in_area(eq1, 'B3', 'C3', scale_factor=1.0)
        self.place_at_grid(labels[0], 'C3', scale_factor=0.8)
        self.play(Write(eq1), Write(labels[0]))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFD700"))
        self.place_in_area(eq2, 'D2', 'E2', scale_factor=1.0)
        self.place_at_grid(labels[1], 'E2', scale_factor=0.8)
        self.play(Write(eq2), Write(labels[1]))
        
        self.place_in_area(eq3, 'D4', 'E4', scale_factor=1.0)
        self.place_at_grid(labels[2], 'E4', scale_factor=0.8)
        self.play(Write(eq3), Write(labels[2]))
        
        triangle = VGroup(
            Line(eq1.get_bottom(), eq2.get_top(), color="#00FFFF"),
            Line(eq2.get_right(), eq3.get_left(), color="#00FFFF"),
            Line(eq3.get_top(), eq1.get_bottom(), color="#00FFFF")
        )
        self.play(Create(triangle))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF4500"))
        self.play(Indicate(eq1), Indicate(eq2), Indicate(eq3))
        self.wait(2)
