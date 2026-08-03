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
        title_text = "The Familiar: Vectors as Geometric Arrows"
        lecture_lines = [
            "Traditionally, we visualize vectors as arrows in space.",
            "An arrow has both a magnitude and a direction.",
            "We add vectors by placing them tip-to-tail.",
            "Multiplying by a scalar stretches or shrinks the arrow.",
            "In geometry, vectors represent physical movement or force."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Origin for our coordinate system is at D2
        origin = self.grid['D2']
        
        # === Animation for Lecture Line 1 ===
        # Traditionally, we visualize vectors as arrows in space.
        self.lecture[0].set_color(WHITE)
        
        # Show axes
        axes = VGroup(
            Line(self.grid['F2'], self.grid['A2'], color=BLUE_E, stroke_width=2), # Y axis
            Line(self.grid['D1'], self.grid['D6'], color=BLUE_E, stroke_width=2)  # X axis
        )
        self.play(Create(axes))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # An arrow has both a magnitude and a direction.
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        
        # Green vector v from (0,0) to (3,2) -> D2 to B5
        v_end = self.grid['B5']
        v_arrow = Arrow(origin, v_end, buff=0, color="#00FF00")
        v_label = MathTex("v", color="#00FF00")
        self.place_at_grid(v_label, "C5", scale_factor=0.8)
        
        self.play(GrowArrow(v_arrow), Write(v_label))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # We add vectors by placing them tip-to-tail.
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        
        # Yellow vector u from (3,2) to (4,3) -> B5 to A6
        u_start = v_end
        u_end = self.grid['A6']
        u_arrow = Arrow(u_start, u_end, buff=0, color="#FFFF00")
        u_label = MathTex("u", color="#FFFF00")
        self.place_at_grid(u_label, "A6", scale_factor=0.8) # Fix: Issue 23
        
        # Sum vector v+u from (0,0) to (4,3) -> D2 to A6
        sum_arrow = Arrow(origin, u_end, buff=0, color=WHITE)
        sum_label = MathTex("v+u", color=WHITE)
        self.place_at_grid(sum_label, "B2", scale_factor=0.8) # Fix: Issue 24
        
        self.play(GrowArrow(u_arrow), Write(u_label))
        self.wait(1)
        self.play(GrowArrow(sum_arrow), Write(sum_label))
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        # Multiplying by a scalar stretches or shrinks the arrow.
        self.play(self.lecture[3].animate.set_color("#FF00FF"))
        
        # Fade out addition elements to focus on scaling
        self.play(
            FadeOut(u_arrow), FadeOut(u_label), 
            FadeOut(sum_arrow), FadeOut(sum_label),
            FadeOut(v_arrow), FadeOut(v_label)
        )
        
        # Magenta vector w at (1, 0.5) -> Start D2, End halfway between D3 and C3
        w_start = origin
        w_end_small = (self.grid['D3'] + self.grid['C3']) / 2
        w_arrow = Arrow(w_start, w_end_small, buff=0, color="#FF00FF")
        w_label = MathTex("v", color="#FF00FF")
        self.place_at_grid(w_label, "E3", scale_factor=0.8)
        
        # Scaled vector 2v at (2, 1) -> Start D2, End C4
        w_end_large = self.grid['C4']
        w_arrow_scaled = Arrow(w_start, w_end_large, buff=0, color="#FF00FF")
        w_label_target = MathTex("2v", color="#FF00FF")
        self.place_at_grid(w_label_target, "B5", scale_factor=0.8) # Fix: Issue 25
        
        self.play(GrowArrow(w_arrow), Write(w_label))
        self.wait(1)
        self.play(
            Transform(w_arrow, w_arrow_scaled),
            Transform(w_label, w_label_target)
        )
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        # In geometry, vectors represent physical movement or force.
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(3)
