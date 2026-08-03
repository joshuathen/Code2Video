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
        # Setup the layout with the specific title and lecture lines
        self.setup_layout(
            "The Dissipation Scale: Where Energy Becomes Heat",
            [
                "At the smallest scales, viscosity becomes dominant.",
                "The Kolmogorov length defines the limit of chaos.",
                "Eddies can no longer exist below this scale.",
                "Kinetic energy finally transforms into internal heat.",
                "Chaos ends where molecular friction takes over."
            ]
        )
        
        # Define colors
        WHITE_COLOR = "#FFFFFF"
        GRAY_COLOR = "#2F4F4F"
        RED_COLOR = "#FF0000"
        HIGHLIGHT_COLOR = YELLOW

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        
        # Create an eddy (spiral)
        # Fix for Issue 37: Increase size using B2-E5 and scale 0.9
        eddy = ParametricFunction(
            lambda t: np.array([0.5 * t * np.cos(4 * PI * t), 0.5 * t * np.sin(4 * PI * t), 0]),
            t_range=[0, 1],
            color=WHITE_COLOR,
            stroke_width=2
        )
        self.place_in_area(eddy, "B2", "E5", scale_factor=0.9)
        
        # Rotation speed tracker
        rotation_speed = ValueTracker(3 * PI)
        eddy.add_updater(lambda m, dt: m.rotate(rotation_speed.get_value() * dt))
        
        self.play(Create(eddy))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_COLOR)
        
        # Kolmogorov scale formula
        formula = MathTex(r"\eta = \left(\frac{\nu^3}{\epsilon}\right)^{1/4}", font_size=32, color=WHITE_COLOR)
        self.place_at_grid(formula, "A4", scale_factor=1.0)
        
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)
        
        # Fix for Issue 38: Larger background rectangle
        bg_rect = Rectangle(
            width=5.5, height=5.5,
            fill_color=GRAY_COLOR,
            fill_opacity=0.6,
            stroke_width=0
        ).set_z_index(-1)
        self.place_in_area(bg_rect, "A1", "F6", scale_factor=1.2)
        
        self.play(FadeIn(bg_rect))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(HIGHLIGHT_COLOR)
        
        # Asset: [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/heat.svg]
        heat_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/heat.svg")
        heat_icon.set_color(RED_COLOR)
        self.place_at_grid(heat_icon, "B1", scale_factor=0.6)
        
        # Slow down, turn red, and show heat
        self.play(
            rotation_speed.animate.set_value(0.5 * PI),
            eddy.animate.set_color(RED_COLOR),
            FadeIn(heat_icon),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(HIGHLIGHT_COLOR)
        
        # Friction stops it
        self.play(
            rotation_speed.animate.set_value(0),
            eddy.animate.set_stroke(width=5),
            heat_icon.animate.scale(1.2),
            run_time=1
        )
        self.wait(2)
