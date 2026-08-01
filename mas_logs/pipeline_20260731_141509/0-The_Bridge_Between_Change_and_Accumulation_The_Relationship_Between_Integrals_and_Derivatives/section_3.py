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

class Section3Scene(TeachingScene):
    def construct(self):
        title = "Defining the Integral: The 'Gluer'"
        lines = [
            "Integrals \"glue\" tiny slices back together.",
            "They calculate the total area under a curve.",
            "This area represents the total accumulation of change.",
            "For Turbo, it's the total distance he traveled.",
            "Integration reverses the slicing process of the derivative."
        ]
        self.setup_layout(title, lines)

        # Colors
        BLUE_CURVE = "#1E90FF"
        PURPLE_RECT = "#8A2BE2"

        # Setup Axes on the right side
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 4, 1],
            axis_config={"include_tip": True, "font_size": 18, "color": WHITE},
            x_length=4.5,
            y_length=3.5,
        ).add_coordinates()
        self.place_in_area(axes, "A1", "F6")
        
        # Labels for axes
        x_label = Text("Time (t)", font_size=16, color=WHITE).next_to(axes.x_axis, DOWN, buff=0.2)
        y_label = Text("Velocity (v)", font_size=16, color=WHITE).next_to(axes.y_axis, LEFT, buff=0.2).rotate(90 * DEGREES)
        
        # Velocity Curve: v(t) = -0.3*(t-2.5)^2 + 3
        curve = axes.plot(lambda t: -0.3 * (t - 2.5)**2 + 3, x_range=[0, 5], color=BLUE_CURVE)
        curve_label = MathTex(r"v(t)", color=BLUE_CURVE, font_size=24)
        # Issue 36: Position curve_label at B6 to avoid overlap
        self.place_at_grid(curve_label, "B6", scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        # Integrals "glue" tiny slices back together.
        self.play(self.lecture[0].animate.set_color(BLUE_CURVE))
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.play(Create(curve), Write(curve_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # They calculate the total area under a curve.
        self.play(self.lecture[1].animate.set_color(PURPLE_RECT))
        
        # Initial coarse rectangles (small slices)
        rects_coarse = axes.get_riemann_rectangles(curve, x_range=[0.5, 4.5], dx=0.5, color=PURPLE_RECT, fill_opacity=0.5)
        self.play(Create(rects_coarse))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # This area represents the total accumulation of change.
        self.play(self.lecture[2].animate.set_color(PURPLE_RECT))
        
        # Finer rectangles to represent "gluing"
        rects_fine = axes.get_riemann_rectangles(curve, x_range=[0.5, 4.5], dx=0.1, color=PURPLE_RECT, fill_opacity=0.7)
        self.play(ReplacementTransform(rects_coarse, rects_fine))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # For Turbo, it's the total distance he traveled.
        self.play(self.lecture[3].animate.set_color(PURPLE_RECT))
        
        # Final smooth area
        area = axes.get_area(curve, x_range=[0.5, 4.5], color=PURPLE_RECT, opacity=0.8)
        distance_label = Text("Distance", font_size=24, color=WHITE)
        # Issue 37: Position distance_label in a larger area (D3-E5)
        self.place_in_area(distance_label, "D3", "E5", scale_factor=0.8)
        
        self.play(
            FadeOut(rects_fine),
            FadeIn(area),
            Write(distance_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Integration reverses the slicing process of the derivative.
        self.play(self.lecture[4].animate.set_color(BLUE_CURVE))
        
        # Visualizing the "reversal" - highlighting the curve again
        self.play(Indicate(curve, color=BLUE_CURVE))
        
        # Symbolic representation of reversal
        integral_formula = MathTex(r"d = \int v(t) dt", color=WHITE, font_size=32)
        # Issue 35: Position integral_formula in area (A3-A5)
        self.place_in_area(integral_formula, "A3", "A5", scale_factor=0.9)
        self.play(Write(integral_formula))
        
        self.wait(2)
