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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Visualizing the Limit Process", [
            "Zoom in on the curve.",
            "Move two points closer together.",
            "Secant line becomes a tangent line."
        ])
        
        # Use provided asset for the curve
        parabola_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/parabola.svg"
        curve = SVGMobject(parabola_asset).set_color(WHITE)
        
        # Setup axes for visual reference if needed or just use grid/curve
        axes = Axes(x_range=[-1, 3], y_range=[-1, 5], axis_config={"include_tip": False})
        
        # Grid area for plot - Fix per VideoCritic/Orchestrator
        self.place_in_area(axes, "B3", "E5", scale_factor=0.5)
        self.place_in_area(curve, "B3", "E5", scale_factor=0.5)
        
        # Points
        p1 = Dot(color=YELLOW)
        p2 = Dot(color=RED)
        
        # Position points per VideoCritic
        self.place_at_grid(p1, "D4", scale_factor=0.7)
        self.place_at_grid(p2, "D5", scale_factor=0.7)
        
        # Secant line
        secant = Line(p1.get_center(), p2.get_center(), color=RED, stroke_width=3)
        self.place_in_area(secant, "C4", "E6", scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_opacity(1)
        self.play(FadeIn(axes), FadeIn(curve), FadeIn(p1), FadeIn(p2))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(GRAY)
        self.lecture[1].set_opacity(1)
        self.add(secant)
        self.play(p2.animate.move_to(p1.get_center() + RIGHT * 0.2), run_time=2)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(GRAY)
        self.lecture[2].set_opacity(1)
        tangent = Line(color=GREEN, stroke_width=3)
        self.place_in_area(tangent, "C4", "E6", scale_factor=0.6)
        self.play(
            FadeOut(secant),
            FadeOut(p2),
            Create(tangent),
            run_time=1.5
        )
        self.wait(2)
