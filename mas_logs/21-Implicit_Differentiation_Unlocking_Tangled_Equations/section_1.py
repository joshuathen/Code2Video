from manim import *

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

class Section1Scene(TeachingScene):
    def construct(self):
        # Setup the scene layout with updated lecture lines from Issue 46
        lecture_lines = [
            "Implicit functions aren't easily solved for y.",
            "We treat y as a function of x.",
            "Differentiate both sides with respect to x."
        ]
        self.setup_layout("The Mystery of the Tangled Curve", lecture_lines)

        # Common coordinate system for visualization
        # Adjusted size to fit B2-F6 area
        axes = Axes(
            x_range=[-6, 6, 2],
            y_range=[-6, 6, 2],
            axis_config={"include_tip": False, "color": GRAY_C},
            x_length=4,
            y_length=4
        )
        self.place_in_area(axes, "B2", "F6")

        # === Animation for Lecture Line 1 ===
        # Colors: Green for explicit starting comparison
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        
        parabola = axes.plot(lambda x: 0.25 * x**2, x_range=[-4, 4], color="#00FF00")
        parabola_label = Text("y = x²", color="#00FF00", font_size=32)
        # Issue 31: Place parabola_label at A2
        self.place_at_grid(parabola_label, "A2", scale_factor=0.8)
        
        # Issue 25: Use drone asset
        drone = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/drone.svg")
        drone.set_color(WHITE).scale(0.3)
        drone.move_to(parabola.get_start())
        
        self.play(Create(axes), Create(parabola), Write(parabola_label))
        self.play(MoveAlongPath(drone, parabola, rate_func=linear, run_time=3))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Colors: Cyan for implicit
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        
        # Create Implicit Circle
        # Scale circle radius to fit axes
        radius_val = 5
        circle = axes.plot_parametric_curve(
            lambda t: np.array([radius_val * np.cos(t), radius_val * np.sin(t), 0]),
            t_range=[0, TAU],
            color="#00FFFF"
        )
        
        # Implicit Equation: x² + y² = 25 
        eq_part1 = Text("x² + ", color="#00FFFF", font_size=36)
        eq_part2 = Text("y", color="#00FFFF", font_size=36)
        eq_part3 = Text("² = 25", color="#00FFFF", font_size=36)
        equation = VGroup(eq_part1, eq_part2, eq_part3).arrange(RIGHT, buff=0.05)
        # Issue 29: Place equation at A4-A6
        self.place_in_area(equation, "A4", "A6", scale_factor=0.7)
        
        # Simple Lock Symbol
        lock_body = Square(side_length=0.2, color=RED, fill_opacity=1)
        lock_arc = Arc(radius=0.1, start_angle=0, angle=PI, color=RED)
        lock_arc.next_to(lock_body, UP, buff=0)
        lock_symbol = VGroup(lock_body, lock_arc).scale(0.8)
        # Position lock over 'y' (eq_part2) in the equation
        lock_symbol.next_to(eq_part2, UP, buff=0.1)

        self.play(
            FadeOut(parabola),
            FadeOut(parabola_label),
            FadeOut(drone),
            Create(circle),
            Write(equation)
        )
        self.play(FadeIn(lock_symbol))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Colors: Yellow for tangent/slope
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        
        # Point (3, 4) on the circle
        p_coords = axes.coords_to_point(3, 4)
        point_on_circle = Dot(p_coords, color=WHITE)
        point_label = Text("(3, 4)", font_size=24, color=WHITE).next_to(point_on_circle, UR, buff=0.1)
        
        # Tangent line at (3, 4) for x² + y² = 25
        # Slope m = -x/y = -3/4
        angle = np.arctan(-3/4)
        tangent_line = Line(
            start=p_coords + 1.2 * np.array([np.cos(angle), np.sin(angle), 0]),
            end=p_coords - 1.2 * np.array([np.cos(angle), np.sin(angle), 0]),
            color="#FFFF00",
            stroke_width=6
        )

        self.play(FadeIn(point_on_circle), Write(point_label))
        self.play(Create(tangent_line))
        
        # Pulse animation for tangent line
        self.play(
            tangent_line.animate.scale(1.2).set_stroke(width=10),
            rate_func=there_and_back,
            run_time=1.5
        )
        self.play(
            tangent_line.animate.scale(1.2).set_stroke(width=10),
            rate_func=there_and_back,
            run_time=1.5
        )
        
        self.wait(2)
