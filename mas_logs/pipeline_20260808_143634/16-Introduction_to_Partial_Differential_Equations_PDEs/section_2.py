from manim import *
import os

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
        self.setup_layout("The Core Anatomy: Variables and Operators", [
            "We denote PDEs as u(x, y, t).",
            "Partial derivatives measure local slopes.",
            "They calculate change along specific axes."
        ])
        
        # === Animation for Lecture Line 1 ===
        # We denote PDEs as u(x, y, t).
        var_tex = MathTex("u(x, y, t)", font_size=48, color=WHITE)
        self.place_at_grid(var_tex, 'A4', scale_factor=1.0)
        self.play(Write(var_tex))
        self.play(self.lecture[0].animate.set_color("#FF6600"))
        self.play(var_tex.animate.set_color("#FF6600"))

        # === Animation for Lecture Line 2 ===
        # Partial derivatives measure local slopes.
        pd_tex = MathTex(r"\frac{\partial u}{\partial x}, \frac{\partial u}{\partial y}", font_size=42, color=WHITE)
        self.place_at_grid(pd_tex, 'B4', scale_factor=0.8)
        self.play(FadeIn(pd_tex))
        self.play(self.lecture[1].animate.set_color("#00CCFF"))
        self.play(pd_tex.animate.set_color("#00CCFF"))

        # === Animation for Lecture Line 3 ===
        # They calculate change along specific axes.
        axes = ThreeDAxes(x_range=[-2, 2], y_range=[-2, 2], z_range=[-1, 1], axis_config={"include_tip": True})
        self.place_in_area(axes, 'C3', 'E5', scale_factor=1.2)
        
        # Using placeholder asset as requested (using SVGMobject for icon)
        # Note: icon svg file does not exist, so creating a simple proxy.
        icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg") if os.path.exists("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg") else Dot(color="#00FFFF")
        self.place_at_grid(icon, 'F6', scale_factor=0.5)

        self.play(Create(axes), FadeIn(icon))
        self.play(self.lecture[2].animate.set_color("#FFFF00"), icon.animate.set_color("#00FFFF"))
        self.wait(1)
