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
        # Fetching context from storyboard
        title = "Prerequisite: The Slope of a Secant Line"
        lines = [
            "A hiker climbs along a curved mountain path.",
            "Connecting two points creates a straight secant line.",
            "Its slope measures the average steepness between points."
        ]
        self.setup_layout(title, lines)

        # Colors defined by hex strings (L008)
        COLOR_PATH = "#FFFFFF"
        COLOR_POINTS = "#00FF00"
        COLOR_SECANT = "#FFFF00"
        COLOR_LABEL = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_PATH))

        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 5, 1],
            x_length=4,
            y_length=4,
            axis_config={
                "include_tip": False, 
                "color": "#FFFFFF", 
                "stroke_width": 2
            }
        )
        
        # Define function explicitly for point calculation and plotting
        curve_func = lambda x: 0.15 * x**2 + 0.3 * x + 0.5
        curve = axes.plot(curve_func, x_range=[0, 4.5], color=COLOR_PATH)
        
        x_a, x_b = 1.0, 3.5
        pos_a = axes.c2p(x_a, curve_func(x_a))
        pos_b = axes.c2p(x_b, curve_func(x_b))
        
        dot_a = Dot(pos_a, color=COLOR_POINTS, radius=0.08)
        dot_b = Dot(pos_b, color=COLOR_POINTS, radius=0.08)
        
        # Using Text instead of MathTex to avoid LaTeX compilation issues (L022)
        label_a = Text("A", color=COLOR_POINTS, font_size=24).next_to(dot_a, LEFT, buff=0.15)
        label_b = Text("B", color=COLOR_POINTS, font_size=24).next_to(dot_b, RIGHT, buff=0.15)
        
        secant_line = Line(pos_a, pos_b, color=COLOR_SECANT, stroke_width=4)

        visual_group = VGroup(axes, curve, dot_a, dot_b, label_a, label_b, secant_line)
        # Resolved Issue 25: Scale visual_group to 0.75 for more space
        self.place_in_area(visual_group, "B2", "F6", scale_factor=0.75)

        self.add(axes)
        self.play(Create(curve), run_time=1.5)
        self.wait(1.0)
        self.play(FadeIn(dot_a, dot_b), Write(label_a), Write(label_b))
        self.play(Create(secant_line))
        self.wait(2.0)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color("#FFFFFF"),
            self.lecture[1].animate.set_color(COLOR_SECANT)
        )
        
        p_a = dot_a.get_center()
        p_b = dot_b.get_center()
        corner = np.array([p_b[0], p_a[1], 0])
        
        brace_dx = BraceBetweenPoints(p_a, corner, color=COLOR_LABEL, direction=DOWN, buff=0.1)
        tex_dx = Text("Δx", color=COLOR_LABEL, font_size=24).next_to(brace_dx, DOWN, buff=0.1)
        
        brace_dy = BraceBetweenPoints(corner, p_b, color=COLOR_LABEL, direction=RIGHT, buff=0.1)
        tex_dy = Text("Δy", color=COLOR_LABEL, font_size=24).next_to(brace_dy, RIGHT, buff=0.1)

        self.play(Create(brace_dx), Write(tex_dx))
        self.play(Create(brace_dy), Write(tex_dy))
        self.wait(2.0)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color("#FFFFFF"),
            self.lecture[2].animate.set_color(COLOR_SECANT)
        )
        
        # Replace formula with Text to bypass LaTeX error (L022)
        formula = Text("m = Δy / Δx", color=COLOR_SECANT, font_size=36)
        # Resolved Issue 24: Moved formula to A5 and scaled to 0.9 to avoid overlap
        self.place_at_grid(formula, "A5", scale_factor=0.9)
        
        self.play(Write(formula))
        self.play(Indicate(formula, color=COLOR_SECANT)) # L004
        self.wait(2.0)
