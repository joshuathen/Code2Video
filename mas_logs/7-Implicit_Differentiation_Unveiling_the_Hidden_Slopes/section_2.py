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
        # Setup layout
        title_text = "Explicit vs. Implicit: The Tangled Rope"
        lecture_lines = [
            "Explicit functions isolate y clearly on one side.",
            "Implicit equations like circles leave variables tangled.",
            "Even tangled curves have slopes at every point."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        explicit_label = Text("Explicit:", font_size=20, color=WHITE)
        # Using Text instead of MathTex to avoid 'latex' FileNotFoundError
        explicit_formula = Text("y = 2x + 1", font_size=24, color=WHITE)
        explicit_group = VGroup(explicit_label, explicit_formula).arrange(DOWN, buff=0.2)
        
        implicit_label = Text("Implicit:", font_size=20, color=WHITE)
        # Using MarkupText for superscripts to avoid 'latex' dependency
        implicit_formula = MarkupText("x<sup>2</sup> + y<sup>2</sup> = 25", font_size=24, color=WHITE)
        implicit_group = VGroup(implicit_label, implicit_formula).arrange(DOWN, buff=0.2)
        
        # Fix 35: Equations centered for better horizontal symmetry
        self.place_in_area(explicit_group, "A2", "A3", scale_factor=0.8)
        # Fix 34: Formula moved to row A only to avoid potential overlap with tangent line below
        self.place_in_area(implicit_group, "A4", "A5", scale_factor=0.8)
        
        self.play(FadeIn(explicit_group), FadeIn(implicit_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second lecture line in Blue
        self.play(self.lecture[1].animate.set_color("#1E90FF"))
        
        # Circle area in the center-bottom of the right side
        circle = Circle(radius=1.3, color="#1E90FF")
        # Fix 36: Scaled to 0.9 to prevent crowding
        self.place_in_area(circle, "C2", "F5", scale_factor=0.9)
        
        # Vertical line in red to show failure of vertical line test
        # Using grid points to ensure precise vertical positioning relative to circle
        v_line = Line(self.grid["B4"], self.grid["F4"], color="#FF4500")
        
        self.play(Create(circle))
        self.play(Create(v_line))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third lecture line in Yellow
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        
        # Point and tangent line in yellow to demonstrate slope exists
        angle = 30 * DEGREES
        # Accessing properties of the positioned circle
        point_on_circle = circle.get_center() + circle.radius * np.array([np.cos(angle), np.sin(angle), 0])
        slope_dot = Dot(point_on_circle, color="#FFFF00")
        
        # Tangent line direction is perpendicular to the radius vector
        tangent_direction = np.array([-np.sin(angle), np.cos(angle), 0])
        tangent_line = Line(
            point_on_circle - tangent_direction * 1.0,
            point_on_circle + tangent_direction * 1.0,
            color="#FFFF00"
        )
        
        self.play(FadeIn(slope_dot))
        self.play(Create(tangent_line))
        self.wait(2)
