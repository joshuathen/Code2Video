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

class Section6Scene(TeachingScene):
    def construct(self):
        # Initialization
        title = "Summary and Geometric Insight"
        lines = [
            "Implicit differentiation unlocks slopes for any complex shape.",
            "We find dy/dx without needing to isolate y first.",
            "From circles to hearts, calculus reveals every slope!"
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.play(self.lecture[0].animate.set_color(TEAL))

        # Create Implicit Shapes
        # 1. Ellipse: x^2/4 + y^2 = 1
        ellipse = ImplicitFunction(
            lambda x, y: x**2 / 4 + y**2 - 1,
            color=TEAL,
            x_range=[-2.1, 2.1],
            y_range=[-1.1, 1.1]
        )
        self.place_at_grid(ellipse, "A2", scale_factor=0.4)

        # 2. Hyperbola: x^2 - y^2 = 1
        hyperbola = ImplicitFunction(
            lambda x, y: x**2 - y**2 - 1,
            color=ORANGE,
            x_range=[-2.5, 2.5],
            y_range=[-2.0, 2.0]
        )
        self.place_at_grid(hyperbola, "A5", scale_factor=0.3)

        # 3. Heart Curve: (x^2 + y^2 - 1)^3 - x^2 * y^3 = 0
        heart = ImplicitFunction(
            lambda x, y: (x**2 + y**2 - 1)**3 - x**2 * y**3,
            color=PINK,
            x_range=[-1.5, 1.5],
            y_range=[-1.2, 1.5]
        )
        self.place_in_area(heart, "B3", "D4", scale_factor=0.5)

        self.play(Create(ellipse), Create(hyperbola), Create(heart), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second line
        self.play(self.lecture[1].animate.set_color(GREEN))

        # Tangent Line Helper
        def get_tangent_at(point, slope, color=YELLOW):
            # dx determines half-length of the tangent segment
            dx = 0.4 / np.sqrt(1 + slope**2)
            dy = dx * slope
            line = Line(
                start=point - np.array([dx, dy, 0]),
                end=point + np.array([dx, dy, 0]),
                color=color,
                stroke_width=6
            )
            return line

        # Calculate absolute positions and tangents based on grid centers and scales
        # Ellipse point: x=1, y=sqrt(3)/2 ~ 0.866. Slope m = -x/(4y) = -1/(4*0.866) ~ -0.288
        e_center = self.grid["A2"]
        e_p1 = e_center + np.array([1 * 0.4, 0.866 * 0.4, 0])
        e_t1 = get_tangent_at(e_p1, -0.288)

        # Hyperbola point: x=1.5, y=sqrt(1.25) ~ 1.118. Slope m = x/y = 1.5/1.118 ~ 1.34
        h_center = self.grid["A5"]
        h_p1 = h_center + np.array([1.5 * 0.3, 1.118 * 0.3, 0])
        h_t1 = get_tangent_at(h_p1, 1.34)

        # Heart center and tangents (scaled by 0.5)
        heart_center = heart.get_center()
        heart_p1 = heart_center + np.array([1 * 0.5, 1 * 0.5, 0])
        heart_t1 = get_tangent_at(heart_p1, -1.33)
        
        heart_p2 = heart_center + np.array([-1 * 0.5, 1 * 0.5, 0])
        heart_t2 = get_tangent_at(heart_p2, 1.33)

        # Flash Tangents
        self.play(ShowPassingFlash(e_t1, time_width=0.5), run_time=1)
        self.play(ShowPassingFlash(h_t1, time_width=0.5), run_time=1)
        self.play(ShowPassingFlash(heart_t1, time_width=0.5), run_time=1)
        
        # Flash multiple at once for effect
        self.play(
            ShowPassingFlash(heart_t1.copy(), time_width=0.5),
            ShowPassingFlash(heart_t2, time_width=0.5),
            run_time=1
        )

        # === Animation for Lecture Line 3 ===
        # Highlight third line
        self.play(self.lecture[2].animate.set_color(YELLOW))

        summary_text = Text("Implicit Differentiation: Any curve, any slope!", font_size=24, color="#FFFF00")
        self.place_in_area(summary_text, "F2", "F5", scale_factor=0.8)
        
        self.play(Write(summary_text))
        self.wait(3)
