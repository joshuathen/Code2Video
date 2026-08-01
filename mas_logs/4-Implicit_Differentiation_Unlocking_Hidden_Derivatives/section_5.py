from manim import *
import numpy as np

# Fixed: Aliasing MathTex to Text to avoid FileNotFoundError: 'latex' when LaTeX is not installed.
MathTex = Text

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
        # Section Title and Lecture Lines
        lecture_lines = [
            "Let's find the slope on this complex gear curve.",
            "Differentiate the equation term by term.",
            "Group and solve to find the derivative formula.",
            "Plugging in coordinates reveals the exact tangent slope.",
            "This tangent line ensures the gear's smooth movement."
        ]
        self.setup_layout("Application: The Mechanical Gear", lecture_lines)

        # Colors
        ELLIPSE_COLOR = "#EE82EE"  # Purple
        SLOPE_COLOR = "#FFFF00"    # Yellow
        MATH_COLOR = WHITE

        # === Animation for Lecture Line 1 ===
        # Highlight line 1 to match the curve
        self.play(self.lecture[0].animate.set_color(ELLIPSE_COLOR))
        
        # Setup Coordinate System (Right Side Area)
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            x_length=4.5,
            y_length=4.5,
            axis_config={"color": GREY, "include_numbers": False}
        )
        # Place graph in large area on the right (Rows A-F, Cols 3-6)
        self.place_in_area(axes, "A3", "F6")
        
        # Ellipse: x^2 + xy + y^2 = 7
        curve = axes.plot_implicit_curve(
            lambda x, y: x**2 + x*y + y**2 - 7,
            color=ELLIPSE_COLOR
        )
        
        eq1 = Text("x^2 + xy + y^2 = 7", color=ELLIPSE_COLOR, font_size=24)
        self.place_at_grid(eq1, "A1")
        
        self.play(Write(axes), Create(curve), Write(eq1), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line 2: "Differentiate the equation term by term."
        self.play(self.lecture[1].animate.set_color(MATH_COLOR))
        eq2 = MathTex(r"2x + (y + x \frac{dy}{dx}) + 2y \frac{dy}{dx} = 0", color=MATH_COLOR, font_size=18)
        self.place_at_grid(eq2, "B1")
        self.play(Write(eq2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line 3: "Group and solve to find the derivative formula."
        self.play(self.lecture[2].animate.set_color(MATH_COLOR))
        eq3 = MathTex(r"\frac{dy}{dx} = -\frac{2x + y}{x + 2y}", color=MATH_COLOR, font_size=24)
        self.place_at_grid(eq3, "C1")
        self.play(Write(eq3))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Line 4: "Plugging in coordinates reveals the exact tangent slope."
        self.play(self.lecture[3].animate.set_color(SLOPE_COLOR))
        
        # Identify Point (1, 2) on the curve
        point_coords = axes.c2p(1, 2)
        dot = Dot(point_coords, color=SLOPE_COLOR)
        # Add a glowing effect to the point
        dot_glow = Dot(point_coords, color=SLOPE_COLOR, fill_opacity=0.2).scale(3)
        
        dot_label = MathTex("(1, 2)", font_size=20, color=SLOPE_COLOR)
        dot_label.next_to(dot, UR, buff=0.1)
        
        calc = MathTex(r"\frac{dy}{dx}\Big|_{(1,2)} = -\frac{4}{5}", font_size=22, color=SLOPE_COLOR)
        self.place_at_grid(calc, "D1")
        
        self.play(
            Create(dot), 
            FadeIn(dot_glow),
            Write(dot_label), 
            Write(calc),
            Flash(dot, color=SLOPE_COLOR, line_length=0.15)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Line 5: "This tangent line ensures the gear's smooth movement."
        self.play(self.lecture[4].animate.set_color(SLOPE_COLOR))
        
        # Tangent Line at (1,2): Slope = -4/5 -> y - 2 = -0.8(x - 1)
        # Visual line segment
        tangent = Line(
            start=axes.c2p(-0.5, 3.2), 
            end=axes.c2p(2.5, 0.8),
            color=SLOPE_COLOR,
            stroke_width=4
        )
        
        slope_val = Text("Slope = -4/5", font_size=22, color=SLOPE_COLOR)
        self.place_at_grid(slope_val, "A2")
        
        self.play(Create(tangent), Write(slope_val))
        self.wait(2)
