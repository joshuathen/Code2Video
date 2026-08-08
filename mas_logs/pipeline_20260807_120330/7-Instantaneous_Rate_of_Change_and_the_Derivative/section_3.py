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
        # Setup the layout with section title and lecture lines
        self.setup_layout(
            "The Paradox of the Single Point",
            [
                "Acceleration creates a curved path on our position graph.",
                "Calculating slope at one point leads to zero over zero.",
                "This undefined result creates a challenging mathematical roadblock."
            ]
        )

        # Colors for matching lecture lines
        color_line1 = BLUE_B
        color_line2 = RED_B
        color_line3 = RED

        # === Animation for Lecture Line 1 ===
        # A curved parabola [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/graph.svg] represents increasing speed over time.
        self.lecture[0].set_color(color_line1)
        
        # Add axes for context - Issue 35/46: Move axes to B2-F6 with scale 0.8
        axes = Axes(
            x_range=[0, 6, 1],
            y_range=[0, 6, 1],
            x_length=4,
            y_length=4,
            tips=True,
            axis_config={"include_numbers": False, "color": GREY_B}
        )
        self.place_in_area(axes, "B2", "F6", scale_factor=0.8)
        
        # Load and place graph asset - Issue 26/46
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/graph.svg
        parabola_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/graph.svg")
        parabola_svg.set_color(color_line1)
        self.place_in_area(parabola_svg, "B2", "F6", scale_factor=0.6)
        # Shift slightly to look good within axes
        parabola_svg.shift(RIGHT * 0.5 + UP * 0.5)

        self.play(Create(axes), run_time=1)
        self.play(DrawBorderThenFill(parabola_svg), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A single point 'A' is highlighted on the curve.
        self.lecture[1].set_color(color_line2)
        
        # Pick a point near the middle of the parabola curve
        point_a_coord = parabola_svg.get_center() + RIGHT * 0.3 + UP * 0.3
        dot_a = Dot(point_a_coord, color=color_line2, radius=0.1)
        label_a = MathTex("A", color=color_line2, font_size=30)
        label_a.next_to(dot_a, UR, buff=0.1)
        
        # Formula 0/0 - Issue 33/46: Scale 0.8 at C5
        formula_00 = MathTex("\\frac{0}{0}", color="#FF0000", font_size=48)
        self.place_at_grid(formula_00, "C5", scale_factor=0.8)

        self.play(FadeIn(dot_a), Write(label_a))
        self.play(Write(formula_00))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The formula '0 / 0' flashes in red (#FF0000) near the point.
        # This undefined result creates a challenging mathematical roadblock.
        self.lecture[2].set_color(color_line3)
        
        # "Warning" Roadblock sign representation - Issue 34/46: Scale 0.7 at B5
        warning_sign = VGroup(
            Triangle(color=color_line3).scale(0.8),
            Text("!", color=color_line3, weight=BOLD).scale(0.8)
        )
        self.place_at_grid(warning_sign, "B5", scale_factor=0.7)

        # Flash the 0/0 and show warning sign
        self.play(
            Indicate(formula_00, color=RED, scale_factor=1.5),
            FadeIn(warning_sign, shift=UP),
            run_time=1.5
        )
        
        # Repeated flash for emphasis
        self.play(Flash(formula_00, color=RED, num_lines=12, flash_radius=0.5))
        self.wait(2)
