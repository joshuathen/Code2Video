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
        # Initial Setup
        lines = [
            "Euler's formula links exponential growth to circular rotation.",
            "Angle theta determines our position on the unit circle.",
            "The real part is cosine of the angle.",
            "The imaginary part is sine of the angle.",
            "This formula bridges algebra and geometry perfectly."
        ]
        self.setup_layout("The Master Formula: e^(iθ) = cos(θ) + i sin(θ)", lines)

        # Define Constants and Coordinate system
        theta_val = PI / 3
        # Issue 34 Fix: Scale factor 0.75 for axes at D4
        axes = Axes(
            x_range=[-1.2, 1.2, 1],
            y_range=[-1.2, 1.2, 1],
            x_length=3.5,
            y_length=3.5,
            axis_config={"include_tip": True, "color": GREY_C}
        )
        self.place_at_grid(axes, "D4", scale_factor=0.75)

        # === Animation for Lecture Line 1 ===
        # Euler's formula links exponential growth to circular rotation.
        # Draw white unit circle and radius
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        unit_circle = Circle(radius=axes.x_axis.get_unit_size(), color=WHITE)
        unit_circle.move_to(axes.c2p(0,0))
        
        radius_line = Line(
            axes.c2p(0, 0),
            axes.c2p(np.cos(theta_val), np.sin(theta_val)),
            color=WHITE,
            stroke_width=4
        )
        
        self.play(Create(unit_circle), Create(radius_line), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Angle theta determines our position on the unit circle.
        # Using green (#00FF00) for theta as per instruction
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#00FF00")
        )
        
        angle_arc = Arc(
            radius=0.4 * axes.x_axis.get_unit_size(),
            start_angle=0,
            angle=theta_val,
            arc_center=axes.c2p(0, 0),
            color="#00FF00"
        )
        theta_label = Text("θ", color="#00FF00", font_size=20)
        # Position label relative to arc
        theta_label.move_to(axes.c2p(0.5 * np.cos(theta_val/2), 0.5 * np.sin(theta_val/2)))
        
        self.play(Create(angle_arc), Write(theta_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The real part is cosine of the angle.
        # Highlight horizontal component in red (#FF0000)
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FF0000")
        )
        
        cos_line = Line(
            axes.c2p(0, 0),
            axes.c2p(np.cos(theta_val), 0),
            color="#FF0000",
            stroke_width=6
        )
        cos_text = Text("cos(θ)", color="#FF0000", font_size=20)
        cos_text.next_to(cos_line, DOWN, buff=0.1)
        
        self.play(Create(cos_line), Write(cos_text))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # The imaginary part is sine of the angle.
        # Highlight vertical component in blue (#0000FF)
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color("#0000FF")
        )
        
        sin_line = Line(
            axes.c2p(np.cos(theta_val), 0),
            axes.c2p(np.cos(theta_val), np.sin(theta_val)),
            color="#0000FF",
            stroke_width=6
        )
        sin_text = Text("i sin(θ)", color="#0000FF", font_size=20)
        sin_text.next_to(sin_line, RIGHT, buff=0.1)
        
        self.play(Create(sin_line), Write(sin_text))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # This formula bridges algebra and geometry perfectly.
        # Label the intersection and Fade in the formula at top
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(YELLOW)
        )
        
        # Label intersection point
        point_dot = Dot(axes.c2p(np.cos(theta_val), np.sin(theta_val)), color=WHITE, radius=0.06)
        point_label = Text("cos(θ) + i sin(θ)", font_size=18, color=WHITE)
        point_label.next_to(point_dot, UR, buff=0.05)
        
        # Issue 33 Fix: Final Formula at Area B2 to B5, scale 0.8
        master_formula = Text("e^(iθ) = cos(θ) + i sin(θ)", font_size=32)
        self.place_in_area(master_formula, "B2", "B5", scale_factor=0.8)
        
        self.play(
            FadeIn(point_dot),
            Write(point_label),
            FadeIn(master_formula, shift=DOWN)
        )
        self.wait(2)
