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

class Section4Scene(TeachingScene):
    def construct(self):
        # Data from shared state
        title_text = "The Coordinate Transformation Formula"
        lecture_lines = [
            "We use a formula to transform between perspectives.",
            "Multiply matrix P by the robot's coordinates.",
            "This calculation maps the point back to standard space.",
            "The result tells us the standard XY coordinates.",
            "We have successfully translated the robot's location."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Visualizing the abstract formula first
        formula = Text("[x]_std = P * [x]_new", color=WHITE, font_size=32)
        # L001/Issue 31: Moving formula to Row A to utilize top space and reduce bottom crowding
        self.place_in_area(formula, "A2", "A5", scale_factor=0.9)
        
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Setup the coordinate planes
        # Standard grid for final reference
        std_grid = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            background_line_style={"stroke_color": BLUE_E, "stroke_opacity": 0.4},
            axis_config={"stroke_color": GREY_B, "stroke_width": 2}
        )
        # Issue 30: Positioning grid in Rows C-F to avoid overlap with calculation and frame edges
        self.place_in_area(std_grid, "C2", "F5", scale_factor=0.7)
        grid_center = std_grid.get_center()
        
        # Create the skewed grid (Robot's perspective) using Matrix P = [[2, -1], [1, 1]]
        skewed_grid = std_grid.copy().apply_matrix([[2, -1], [1, 1]], about_point=grid_center)
        skewed_grid.set_color(BLUE_C)
        
        # Vector x_new = (1, 1) in the skewed basis
        dot_pos = skewed_grid.c2p(1, 1)
        dot = Dot(dot_pos, color=YELLOW)
        dot_label = Text("[x]_new = (1, 1)", color=YELLOW, font_size=18)
        dot_label.next_to(dot, UR, buff=0.1) # L003: Label close to object
        
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.play(Create(skewed_grid))
        self.play(FadeIn(dot), Write(dot_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Showing the concrete calculation
        calc = Text(
            "[[2, -1], [1, 1]] * [1, 1] = [1, 2]",
            color=WHITE, font_size=24
        )
        # Issue 32: Place calculation in Row B to keep it distinct from formula and grid
        self.place_in_area(calc, "B2", "B5", scale_factor=0.9)
        
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.play(Write(calc))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Transforming perspectives back to standard XY
        new_dot_label = Text("[x]_std = (1, 2)", color=YELLOW, font_size=18)
        new_dot_label.next_to(dot, UR, buff=0.1)
        
        self.play(self.lecture[3].animate.set_color(YELLOW))
        self.play(
            ReplacementTransform(skewed_grid, std_grid),
            ReplacementTransform(dot_label, new_dot_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Highlight the final result
        arrow_start = std_grid.c2p(2.5, -0.5) 
        arrow = Arrow(
            start=arrow_start, 
            end=dot.get_center(), 
            color=RED, 
            buff=0.1
        )
        
        self.play(self.lecture[4].animate.set_color(YELLOW))
        self.play(Create(arrow))
        self.play(Indicate(dot))
        self.wait(2)
