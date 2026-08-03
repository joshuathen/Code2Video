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

class Section7Scene(TeachingScene):
    def construct(self):
        # Configuration
        base_color = BLUE_C
        exp_color = RED_C
        res_color = GREEN_C
        triangle_color = WHITE
        box_color = "#A9A9A9"

        lecture_lines = [
            "The triangle reveals hidden mathematical symmetry.",
            "One static triangle covers all three operations.",
            "Simply cover one corner to create a problem.",
            "Powers, roots, and logs are now one story.",
            "Math is simpler when we see the connections."
        ]

        self.setup_layout("Synthesis: The Beauty of Symmetry", lecture_lines)

        # Pre-define positions for the triangle vertices
        top_pos = self.grid["B4"]
        bl_pos = self.grid["E2"]
        br_pos = self.grid["E6"]

        # === Animation for Lecture Line 1 ===
        # "The triangle reveals hidden mathematical symmetry."
        self.lecture[0].set_color(YELLOW)
        
        triangle = Polygon(top_pos, bl_pos, br_pos, color=triangle_color, stroke_width=4)
        
        # 2 (Base - BL), 10 (Exponent - Top), 1024 (Result - BR)
        base_val = MathTex("2", color=base_color)
        exp_val = MathTex("10", color=exp_color)
        res_val = MathTex("1024", color=res_color)

        # Positioning labels slightly offset from vertices
        self.place_at_grid(base_val, "E2", scale_factor=1.2)
        base_val.shift(LEFT * 0.4 + DOWN * 0.4)
        
        self.place_at_grid(exp_val, "B4", scale_factor=1.2)
        exp_val.shift(UP * 0.4)
        
        # Fix for Issue 39: Scale factor set to 0.85 for '1024'
        self.place_at_grid(res_val, "E6", scale_factor=0.85)
        res_val.shift(RIGHT * 0.5 + DOWN * 0.4)

        self.play(Create(triangle), run_time=1)
        self.play(Write(base_val), Write(exp_val), Write(res_val), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "One static triangle covers all three operations."
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Highlight operations implicitly by showing the triangle is the same for all
        self.play(triangle.animate.set_stroke(width=6), run_time=0.5)
        self.play(triangle.animate.set_stroke(width=4), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Simply cover one corner to create a problem."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # Semi-transparent gray box
        cover_box = Square(side_length=1.2, fill_color=box_color, fill_opacity=0.8, stroke_opacity=0)
        
        # Cycle covering: Result -> Base -> Exponent
        # 1. Cover Result (1024)
        cover_box.move_to(res_val.get_center())
        self.play(FadeIn(cover_box), run_time=0.5)
        self.wait(0.5)
        
        # 2. Move to Base (2)
        self.play(cover_box.animate.move_to(base_val.get_center()), run_time=1)
        self.wait(0.5)
        
        # 3. Move to Exponent (10)
        self.play(cover_box.animate.move_to(exp_val.get_center()), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "Powers, roots, and logs are now one story."
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)

        # Remove box and glow
        self.play(FadeOut(cover_box), run_time=0.5)
        
        glow_circles = VGroup(
            Circle(radius=0.5, color=base_color).move_to(base_val),
            Circle(radius=0.5, color=exp_color).move_to(exp_val),
            Circle(radius=0.7, color=res_color).move_to(res_val)
        ).set_stroke(width=2, opacity=0.5)
        
        self.play(
            LaggedStart(
                *[Indicate(m, color=m.get_color()) for m in [base_val, exp_val, res_val]],
                FadeIn(glow_circles),
                lag_ratio=0.2
            ),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "Math is simpler when we see the connections."
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)

        # Scale up triangle slightly
        full_group = VGroup(triangle, base_val, exp_val, res_val, glow_circles)
        
        # Traditional symbols
        root_sym = MathTex(r"\sqrt{\ }", color=WHITE)
        log_sym = MathTex(r"\log", color=WHITE)
        
        # Fix for Issue 37: root_sym to C3 with scale 0.8
        self.place_at_grid(root_sym, "C3", scale_factor=0.8)
        # Fix for Issue 38: log_sym to C5 with scale 0.8
        self.place_at_grid(log_sym, "C5", scale_factor=0.8)

        self.play(
            full_group.animate.scale(1.1),
            FadeIn(root_sym, shift=UP),
            FadeIn(log_sym, shift=UP),
            run_time=1.5
        )
        self.wait(1)
        
        # Dissolve symbols into the triangle
        self.play(
            root_sym.animate.scale(0.1).move_to(triangle.get_center()).set_opacity(0),
            log_sym.animate.scale(0.1).move_to(triangle.get_center()).set_opacity(0),
            triangle.animate.set_stroke(color=YELLOW, width=8),
            run_time=1.5
        )
        self.play(triangle.animate.set_stroke(color=WHITE, width=4), run_time=0.5)
        
        self.wait(2)
