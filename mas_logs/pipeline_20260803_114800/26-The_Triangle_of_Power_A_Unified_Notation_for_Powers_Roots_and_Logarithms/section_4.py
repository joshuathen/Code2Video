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
        # Data
        title_text = "Operation 1: The Power (Top-Down View)"
        lecture_lines = [
            "To find a power, look from base to exponent.",
            "Five raised to the power of two equals twenty-five.",
            "The result flows naturally to the bottom-right corner."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Triangle Vertices (Visual Anchor System)
        # Updated per Issues 29, 30, 31 to avoid lecture overlap and maintain symmetry
        v_top = self.grid["B4"] # Exponent (Moved from B3)
        v_bl = self.grid["E2"]  # Base (Moved from E1)
        v_br = self.grid["E6"]  # Result (Moved from E5)
        
        # Triangle Mobjects
        triangle_lines = VGroup(
            Line(v_bl, v_top, color=WHITE),
            Line(v_top, v_br, color=WHITE),
            Line(v_br, v_bl, color=WHITE)
        )
        
        # Triangle Values
        base_val = MathTex("5", color="#0000FF")  # blue
        self.place_at_grid(base_val, "E2", scale_factor=1.2)
        
        exp_val = MathTex("2", color="#00FF00")   # green
        self.place_at_grid(exp_val, "B4", scale_factor=1.2)
        
        res_val = MathTex("25", color="#FF0000")  # red
        self.place_at_grid(res_val, "E6", scale_factor=1.2)
        
        # === Animation for Lecture Line 1 ===
        # "To find a power, look from base to exponent."
        # Animation: Show the triangle with '5' (blue) at bottom-left and '2' (green) at the top.
        self.play(self.lecture[0].animate.set_color("#0000FF")) # Match Base color
        self.play(
            Create(triangle_lines),
            FadeIn(base_val),
            FadeIn(exp_val),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Five raised to the power of two equals twenty-five."
        # Animation: Animate a yellow glowing arrow flowing from the Base and Exponent toward Result.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FFFF00") # Match arrow color
        )
        
        arrow_base_to_res = Arrow(v_bl, v_br, color="#FFFF00", buff=0.4)
        arrow_exp_to_res = Arrow(v_top, v_br, color="#FFFF00", buff=0.4)
        
        self.play(
            GrowArrow(arrow_base_to_res),
            GrowArrow(arrow_exp_to_res),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "The result flows naturally to the bottom-right corner."
        # Animation: Reveal '25' (red) at the bottom-right vertex with a scaling and flash effect.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FF0000") # Match result color
        )
        
        # Scaling reveal
        res_val.scale(0.2)
        self.play(
            res_val.animate.scale(5), # 0.2 * 5 = 1.0 (returns to original scale factor)
            FadeIn(res_val),
            Flash(v_br, color="#FF0000", flash_radius=0.5, line_length=0.3),
            run_time=1.5
        )
        # Extra little pulse
        self.play(
            res_val.animate.scale(1.2),
            rate_func=there_and_back,
            run_time=0.8
        )
        
        self.wait(2)
        
        # Final color reset
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
