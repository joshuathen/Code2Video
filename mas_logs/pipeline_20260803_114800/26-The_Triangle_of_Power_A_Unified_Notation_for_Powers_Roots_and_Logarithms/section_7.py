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
        # Lecture Data
        lecture_lines = [
            "The Triangle of Power reveals the harmony in math.",
            "No more switching between confusing symbols and rules.",
            "One mental map simplifies your mathematical journey."
        ]
        self.setup_layout("Summary: The Unified Harmony", lecture_lines)
        
        # Grid positions for the Triangle
        p_base = self.grid["D2"]
        p_exp = (self.grid["B3"] + self.grid["B4"]) / 2
        p_res = self.grid["D5"]
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        triangle = Polygon(p_base, p_exp, p_res, color=WHITE, stroke_width=4)
        val_base = Text("2", font_size=32, color=BLUE)
        val_exp = Text("3", font_size=32, color=RED)
        val_res = Text("8", font_size=32, color=GREEN)
        
        # Position labels near vertices
        val_base.move_to(p_base + LEFT * 0.5 + DOWN * 0.3)
        val_exp.move_to(p_exp + UP * 0.5)
        val_res.move_to(p_res + RIGHT * 0.5 + DOWN * 0.3)
        
        triangle_group = VGroup(triangle, val_base, val_exp, val_res)
        
        self.play(Create(triangle), Write(val_base), Write(val_exp), Write(val_res))
        self.play(
            Rotate(triangle_group, angle=TAU, about_point=triangle.get_center()),
            run_time=4,
            rate_func=linear
        )
        
        self.play(Flash(p_base, color=BLUE), run_time=0.5)
        self.play(Flash(p_exp, color=RED), run_time=0.5)
        self.play(Flash(p_res, color=GREEN), run_time=0.5)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(ORANGE)
        
        # Traditional notation drifting and fading
        sym_pow = MathTex("x^y", color=GRAY)
        sym_root = MathTex("\\sqrt[y]{x}", color=GRAY)
        sym_log = MathTex("\\log_x y", color=GRAY)
        
        # Fixed positions based on issues 34 and 36
        self.place_at_grid(sym_pow, "A4", scale_factor=0.7)
        self.place_at_grid(sym_root, "F3", scale_factor=0.7)
        self.place_at_grid(sym_log, "F6", scale_factor=0.7)
        
        self.play(FadeIn(sym_pow), FadeIn(sym_root), FadeIn(sym_log))
        self.wait(0.5)
        
        self.play(
            sym_pow.animate.shift(UP * 1.5 + LEFT * 1).set_opacity(0),
            sym_root.animate.shift(LEFT * 1.5 + DOWN * 1).set_opacity(0),
            sym_log.animate.shift(RIGHT * 1.5 + DOWN * 1).set_opacity(0),
            run_time=2
        )
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(TEAL)
        
        # Glowing effect for Triangle
        glow = triangle.copy().set_stroke(WHITE, width=15, opacity=0.8)
        self.play(
            triangle.animate.set_stroke(WHITE, width=10),
            FadeIn(glow),
            run_time=0.5
        )
        self.play(
            glow.animate.scale(1.4).set_opacity(0),
            run_time=1.0,
            rate_func=rate_functions.ease_out_quad
        )
        self.remove(glow)
        
        # Character "Leo" (Circle)
        # Fixed position and scale based on issue 35
        leo = Circle(radius=0.2, color="#FFFFE0", fill_opacity=1.0)
        self.place_at_grid(leo, "E3", scale_factor=0.5)
        
        self.play(FadeIn(leo))
        
        # Jumping across the scene
        for _ in range(3):
            self.play(leo.animate.shift(UP * 0.6), run_time=0.3, rate_func=rate_functions.ease_out_sine)
            self.play(leo.animate.shift(DOWN * 0.6), run_time=0.3, rate_func=rate_functions.ease_in_sine)
            
        self.wait(2)
        self.lecture[2].set_color(WHITE)
        self.wait(1)
