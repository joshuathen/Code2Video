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
        lecture_lines = [
            "For large masses, the rotation angle becomes very small.",
            "The number of collisions is pi divided by the angle.",
            "This explains why the digits of pi appear."
        ]
        self.setup_layout("The Mathematical Bridge: Small Angles and Pi", lecture_lines)
        
        # Load asset
        mass_asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/mass.svg"
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#ADD8E6"))
        
        # Formula: theta approx sqrt(m/M)
        theta_formula = MathTex(r"\theta \approx \sqrt{\frac{m}{M}}", color="#ADD8E6")
        self.place_at_grid(theta_formula, "B5", scale_factor=1.2) # Fixed position from Issue 34
        
        # Asset for line 1
        icon1 = SVGMobject(mass_asset_path, fill_color="#ADD8E6")
        self.place_at_grid(icon1, "B3", scale_factor=0.6)
        
        self.play(Write(theta_formula), FadeIn(icon1))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#ADD8E6")
        )
        
        # N = Pi / theta near a circle arc
        # First, draw an arc representing the phase space journey
        arc = Arc(radius=1.5, start_angle=0, angle=PI, color=BLUE_B)
        self.place_in_area(arc, "C2", "E4", scale_factor=1.0)
        
        n_formula = MathTex(r"N \approx \frac{\pi}{\theta}", color="#ADD8E6")
        self.place_at_grid(n_formula, "D5", scale_factor=1.2) # Fixed position from Issue 35
        
        self.play(Create(arc))
        self.play(Write(n_formula))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FFFFFF")
        )
        
        # Result N = 314 for M = 10,000
        result_text = MathTex(r"M = 10,000 \cdot m \implies N = 314", color="#FFFFFF")
        self.place_in_area(result_text, "F1", "F6", scale_factor=1.0) # Fixed position from Issue 36
        
        # Asset for line 3
        icon2 = SVGMobject(mass_asset_path, fill_color=WHITE)
        self.place_at_grid(icon2, "E5", scale_factor=0.6)
        
        self.play(Write(result_text), FadeIn(icon2))
        self.wait(3)

        # Cleanup
        self.play(
            FadeOut(theta_formula),
            FadeOut(icon1),
            FadeOut(arc),
            FadeOut(n_formula),
            FadeOut(result_text),
            FadeOut(icon2),
            self.lecture[2].animate.set_color(WHITE)
        )
        self.wait(1)
