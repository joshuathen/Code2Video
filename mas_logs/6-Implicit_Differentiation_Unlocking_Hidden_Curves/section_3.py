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
        # Initializing the layout with the lecture content
        lecture_lines = [
            "Apply the derivative operator to every single term.",
            "Power rule handles x-squared normally, giving two x.",
            "For y-squared, the chain rule adds dy dx.",
            "The derivative of a constant, twenty-five, is zero.",
            "Our equation now links x, y, and slope."
        ]
        self.setup_layout("Step 1: The 'Differentiate Both Sides' Blitz", lecture_lines)

        # Colors
        COLOR_X = "#58C4DD"  # Blue
        COLOR_Y = "#FF0000"  # Red

        # === Animation for Lecture Line 1 ===
        # Equation: x^2 + y^2 = 25
        # Replacing MathTex with VGroup of Text to avoid LaTeX dependency error
        eq1 = VGroup(
            Text("x²"), Text("+"), Text("y²"), Text("="), Text("25")
        ).arrange(RIGHT, buff=0.15)
        self.place_in_area(eq1, "A1", "C6", scale_factor=1.5)
        
        # Operator Equation: d/dx(x^2) + d/dx(y^2) = d/dx(25)
        eq2 = VGroup(
            Text("d/dx(x²)"), 
            Text("+"), 
            Text("d/dx(y²)"), 
            Text("="), 
            Text("d/dx(25)")
        ).arrange(RIGHT, buff=0.15)
        self.place_in_area(eq2, "A1", "C6", scale_factor=1.2)

        self.lecture[0].set_color(WHITE)
        self.play(Write(eq1))
        self.wait(1)
        self.play(ReplacementTransform(eq1, eq2))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Power rule handles x-squared normally, giving two x.
        self.lecture[1].set_color(COLOR_X)
        
        # Final simplified terms (pre-calculated for alignment)
        eq3 = VGroup(
            Text("2x"),           # 0
            Text("+"),            # 1
            Text("2y"),           # 2
            Text("·"),            # 3
            Text("dy/dx"),        # 4
            Text("="),            # 5
            Text("0")             # 6
        ).arrange(RIGHT, buff=0.15)
        eq3[0].set_color(COLOR_X)
        eq3[4].set_color(COLOR_Y)
        self.place_in_area(eq3, "D1", "F6", scale_factor=1.5)

        # Animation: transform d/dx(x^2) into 2x
        term_x = eq3[0].copy().move_to(eq2[0].get_center())
        self.play(
            Transform(eq2[0], term_x),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # For y-squared, the chain rule adds dy dx.
        self.lecture[2].set_color(COLOR_Y)
        
        # Animation: transform d/dx(y^2) into 2y * dy/dx
        term_y_group = VGroup(eq3[2], eq3[3], eq3[4]).copy().move_to(eq2[2].get_center())
        self.play(
            Transform(eq2[2], term_y_group),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # The derivative of a constant, twenty-five, is zero.
        self.lecture[3].set_color(WHITE)
        
        # Animation: transform d/dx(25) into 0
        term_zero = eq3[6].copy().move_to(eq2[4].get_center())
        self.play(
            Transform(eq2[4], term_zero),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Our equation now links x, y, and slope.
        self.lecture[4].set_color(WHITE)
        
        # Final Cleanup: Transition from the messy intermediate to a clean final equation
        current_eq = VGroup(eq2[0], eq2[1], eq2[2], eq2[3], eq2[4])
        self.play(
            ReplacementTransform(current_eq, eq3),
            run_time=1.5
        )
        self.play(Indicate(eq3))
        self.wait(2)
