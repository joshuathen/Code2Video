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
        lecture_lines = ["Null space maps non-zero inputs to origin.", "It highlights ambiguity in the transformation.", "Information is lost along null space directions."]
        self.setup_layout("Null Space: The 'Ghost' Inputs", lecture_lines)
        
        # Setup coordinate system
        axes = Axes(x_range=[-3, 3], y_range=[-3, 3], axis_config={"include_tip": True}).scale(0.6)
        self.place_in_area(axes, "B3", "E5", scale_factor=0.9)
        self.add(axes)

        # Assets
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg]
        # Since icon/none.svg is a placeholder, we use a default shape/image if it doesn't exist
        # or treat it as a generic indicator.
        def get_placeholder(color=WHITE):
            return Dot(color=color)

        # === Animation for Lecture Line 1 ===
        # Draw a vector that maps to the zero vector.
        v_in = Vector(RIGHT + UP, color=YELLOW)
        self.place_at_grid(v_in, "B4", scale_factor=0.8)
        
        v_label = MathTex("x", color=YELLOW).next_to(v_in.get_end(), UP)
        
        self.play(Create(v_in), Write(v_label))
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(v_in.animate.set_opacity(0), v_label.animate.set_opacity(0))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight the set of all such vectors.
        line = Line(start=axes.c2p(-2, -2), end=axes.c2p(2, 2), color=BLUE)
        self.play(Create(line))
        self.play(self.lecture[1].animate.set_color(BLUE))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Label this set as the null space.
        ns_label = Text("Null Space", font_size=20, color=BLUE)
        self.place_at_grid(ns_label, "D5", scale_factor=0.7)
        
        self.play(Write(ns_label))
        self.play(self.lecture[2].animate.set_color(BLUE))
        self.wait(2)
