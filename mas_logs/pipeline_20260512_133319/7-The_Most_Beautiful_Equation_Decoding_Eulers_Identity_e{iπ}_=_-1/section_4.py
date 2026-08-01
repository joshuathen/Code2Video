from manim import *
import numpy as np
import pathlib

# Fix for Manim CE KeyError: 'iπ'
# This intercept sanitizes input_file paths containing curly braces.
from manim import config as global_config
try:
    _raw_path = str(global_config.input_file)
    if "{" in _raw_path:
        sanitized_path = _raw_path.replace("{", "").replace("}", "")
        global_config.input_file = pathlib.Path(sanitized_path)
except Exception:
    pass

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
        # Setup title and lecture lines
        lecture_lines = [
            "What happens when growth is powered by i?",
            "Usually, growth scales a value outward along a line.",
            "With i, growth is always perpendicular to current position.",
            "This sideways growth creates a perfect circular orbit.",
            "Euler’s formula describes this rotation around the origin."
        ]
        self.setup_layout("The Synthesis: Sideways Growth", lecture_lines)

        # Colors
        COLOR_OUTWARD = BLUE_B
        COLOR_PERP = "#FF00FF" # Purple
        COLOR_CIRCLE = WHITE
        COLOR_FORMULA = YELLOW_B

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        
        # Create Complex Plane Axes
        # Using a square range to ensure circular motion looks circular
        axes = Axes(
            x_range=[-1.5, 1.5, 0.5],
            y_range=[-1.5, 1.5, 0.5],
            x_length=4,
            y_length=4,
            axis_config={"include_tip": True, "color": GREY_D}
        )
        
        # Grid placement per issue 49 (orbit_axes_group B2-F6)
        orbit_axes_group = VGroup(axes)
        self.place_in_area(orbit_axes_group, 'B2', 'F6', scale_factor=0.9)
        
        self.play(Create(axes), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(COLOR_OUTWARD)
        
        # Radius vector from origin to (1, 0)
        radius_vector = Arrow(
            start=axes.c2p(0, 0),
            end=axes.c2p(1, 0),
            buff=0,
            color=WHITE,
            stroke_width=4
        )
        
        # Outward growth vector (velocity for normal e^t)
        outward_velocity = Arrow(
            start=axes.c2p(1, 0),
            end=axes.c2p(1.5, 0),
            buff=0,
            color=COLOR_OUTWARD,
            stroke_width=6
        )
        
        self.play(GrowArrow(radius_vector))
        self.play(GrowArrow(outward_velocity))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(COLOR_PERP)
        
        # Velocity vector perpendicular (pointing up at (1,0))
        perp_velocity = Arrow(
            start=axes.c2p(1, 0),
            end=axes.c2p(1, 0.5),
            buff=0,
            color=COLOR_PERP,
            stroke_width=6
        )
        
        # Label for sideways growth
        growth_label = Text("Sideways Growth", font_size=24, color=COLOR_PERP)
        self.place_at_grid(growth_label, 'B6', scale_factor=0.8) # Position per issue 49
        
        self.play(
            ReplacementTransform(outward_velocity, perp_velocity),
            Write(growth_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(COLOR_CIRCLE)
        
        # Group vectors to rotate them together
        rotating_group = VGroup(radius_vector, perp_velocity)
        
        # Trace for the circle
        circle_path = Arc(radius=axes.get_x_unit_size(), start_angle=0, angle=TAU, color=COLOR_CIRCLE)
        circle_path.move_to(axes.c2p(0,0)) # Align with origin
        
        # Manual rotation for performance and to match tracing
        # We'll do a full rotation
        self.play(
            Rotate(rotating_group, angle=TAU, about_point=axes.c2p(0, 0)),
            Create(circle_path),
            run_time=4,
            rate_func=linear
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(COLOR_FORMULA)
        
        # Formula: e^{it}
        # Building manually to avoid Tex/MathTex
        e_text = Text("e", font_size=40, color=COLOR_FORMULA)
        it_text = Text("it", font_size=28, color=COLOR_FORMULA).next_to(e_text.get_top(), RIGHT, buff=0.05).shift(DOWN*0.1)
        euler_formula = VGroup(e_text, it_text)
        
        # Placement per issue 49 (grid A3)
        self.place_at_grid(euler_formula, 'A3', scale_factor=1.0)
        
        self.play(Write(euler_formula))
        self.play(Indicate(euler_formula, color=COLOR_FORMULA))
        
        self.wait(2)
