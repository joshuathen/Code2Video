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

class Section5Scene(TeachingScene):
    def construct(self):
        # Title and Lecture Lines
        title = "Abstract Example 2: The Polynomial Kingdom"
        lines = [
            "Polynomials can be viewed as vectors in higher dimensions.",
            "Coefficients represent coordinates in an abstract coordinate system.",
            "Adding polynomials simply sums their corresponding coefficients.",
            "The degree of the polynomial defines the space's dimension.",
            "This maps algebraic objects onto geometric structures."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # Line 1: Polynomials can be viewed as vectors in higher dimensions.
        # Animation: Display the expression '3 + 2x + 5x^2' in #FFFFFF.
        self.lecture[0].set_color(WHITE)
        poly = MathTex("3", "+", "2", "x", "+", "5", "x^2", color=WHITE)
        self.place_at_grid(poly, "B3", scale_factor=1.2)
        self.play(Write(poly))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line 2: Coefficients represent coordinates in an abstract coordinate system.
        # Animation: Highlight the coefficients 3, 2, and 5 in #FFFF00.
        self.lecture[1].set_color("#FFFF00")
        
        # MathTex index reference: 0:'3', 2:'2', 5:'5'
        c1 = poly[0]
        c2 = poly[2]
        c3 = poly[5]
        
        self.play(
            c1.animate.set_color("#FFFF00"),
            c2.animate.set_color("#FFFF00"),
            c3.animate.set_color("#FFFF00")
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line 3: Adding polynomials simply sums their corresponding coefficients.
        # Animation: Animate the coefficients moving into a #00FF00 coordinate tuple (3, 2, 5). Label it 'Coordinate Vector'.
        self.lecture[2].set_color("#00FF00")
        
        # Tuple components (brackets and commas)
        coord_tuple = MathTex("(", "3", ",", "2", ",", "5", ")", color="#00FF00")
        self.place_at_grid(coord_tuple, "C3", scale_factor=1.2)
        
        # Background elements (brackets and commas)
        tuple_bg = VGroup(coord_tuple[0], coord_tuple[2], coord_tuple[4], coord_tuple[6])
        
        # Placeholder/Target positions for the numbers
        t1 = coord_tuple[1]
        t2 = coord_tuple[3]
        t3 = coord_tuple[5]
        t1.set_opacity(0)
        t2.set_opacity(0)
        t3.set_opacity(0)

        label = Text("Coordinate Vector", font_size=24, color="#00FF00")
        self.place_at_grid(label, "C5", scale_factor=0.8)
        
        self.play(Write(tuple_bg), Write(label))
        
        c1_copy = c1.copy()
        c2_copy = c2.copy()
        c3_copy = c3.copy()
        
        self.play(
            c1_copy.animate.move_to(t1).set_color("#00FF00"),
            c2_copy.animate.move_to(t2).set_color("#00FF00"),
            c3_copy.animate.move_to(t3).set_color("#00FF00"),
            run_time=2
        )
        
        # finalize tuple visibility
        t1.set_opacity(1)
        t2.set_opacity(1)
        t3.set_opacity(1)
        self.remove(c1_copy, c2_copy, c3_copy)
        self.add(coord_tuple)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Line 4: The degree of the polynomial defines the space's dimension.
        # Animation: Create a simple #FFFFFF 3D representation using three axes.
        self.lecture[3].set_color(WHITE)
        
        origin = self.grid["E3"]
        axis_x = Arrow(start=origin, end=origin + RIGHT*1.5, color=WHITE, buff=0)
        axis_y = Arrow(start=origin, end=origin + UP*1.5, color=WHITE, buff=0)
        # Simplified Z axis in 2D perspective
        axis_z = Arrow(start=origin, end=origin + np.array([-0.6, -0.6, 0]), color=WHITE, buff=0)
        
        lab_1 = MathTex("1", color=WHITE, font_size=20).next_to(axis_x, RIGHT, buff=0.1)
        lab_x = MathTex("x", color=WHITE, font_size=20).next_to(axis_y, UP, buff=0.1)
        lab_x2 = MathTex("x^2", color=WHITE, font_size=20).next_to(axis_z, DL, buff=0.1)
        
        axes_group = VGroup(axis_x, axis_y, axis_z, lab_1, lab_x, lab_x2)
        
        self.play(Create(axes_group))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Line 5: This maps algebraic objects onto geometric structures.
        # Animation: Plot the point (3, 2, 5) with a #FF00FF dot.
        self.lecture[4].set_color("#FF00FF")
        
        # Scaling: axes represent range [0, 6], physical length 1.5
        unit_scale = 1.5 / 6.0
        vec_x = RIGHT * 3 * unit_scale
        vec_y = UP * 2 * unit_scale
        vec_z = np.array([-0.6, -0.6, 0]) * (5/6.0) # Using the proportion of the arrow length
        
        point_pos = origin + vec_x + vec_y + vec_z
        dot = Dot(point=point_pos, color="#FF00FF", radius=0.08)
        dot_label = MathTex("(3, 2, 5)", color="#FF00FF", font_size=20).next_to(dot, UR, buff=0.05)
        
        # Projection lines for 3D effect clarity
        line_x = DashedLine(origin, origin + vec_x, color=GRAY, stroke_width=1)
        line_xy = DashedLine(origin + vec_x, origin + vec_x + vec_y, color=GRAY, stroke_width=1)
        line_xyz = DashedLine(origin + vec_x + vec_y, point_pos, color=GRAY, stroke_width=1)
        
        self.play(
            Create(line_x),
            Create(line_xy),
            Create(line_xyz),
            FadeIn(dot, scale=0.5),
            Write(dot_label)
        )
        self.wait(2)
