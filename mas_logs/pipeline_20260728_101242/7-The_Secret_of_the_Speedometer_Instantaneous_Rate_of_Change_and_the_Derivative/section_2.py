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

class Section2Scene(TeachingScene):
    def construct(self):
        # Data from storyboard
        title_text = "Prerequisite: The Slope of a Secant Line"
        lecture_lines = [
            "- On a graph, average speed is a secant line.",
            "- It connects two points to find the slope.",
            "- Slope equals the change in y over change in x."
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # Colors from storyboard
        COLOR_CURVE = "#3498DB"
        COLOR_POINTS = "#F1C40F"
        COLOR_SECANT = "#E74C3C"

        # === Animation for Lecture Line 1 ===
        # Draw a smooth curve representing f(x) = x^2 in #3498DB.
        self.lecture[0].set_color(COLOR_CURVE)
        
        # Create axes on the right side
        # Area A1 to E6 (Fix for Issue 24)
        axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 10, 2],
            x_length=4.5,
            y_length=4.5,
            axis_config={"include_tip": True, "font_size": 20}
        ).add_coordinates()
        self.place_in_area(axes, 'A1', 'E6', scale_factor=0.8)
        
        curve = axes.plot(lambda x: x**2, x_range=[0, 3.2], color=COLOR_CURVE)
        curve_label = MathTex("f(x) = x^2", font_size=24, color=COLOR_CURVE)
        curve_label.next_to(curve.point_from_proportion(0.8), UR, buff=0.1)

        self.play(Create(axes), Create(curve), Write(curve_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Plot and label Point A at (1,1) and Point B at (3,9) in #F1C40F.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_POINTS)
        
        point_a_coords = axes.c2p(1, 1)
        point_b_coords = axes.c2p(3, 9)
        
        dot_a = Dot(point_a_coords, color=COLOR_POINTS)
        dot_b = Dot(point_b_coords, color=COLOR_POINTS)
        
        label_a = MathTex("A(1,1)", font_size=20, color=COLOR_POINTS).next_to(dot_a, DL, buff=0.1)
        label_b = MathTex("B(3,9)", font_size=20, color=COLOR_POINTS).next_to(dot_b, UL, buff=0.1)
        
        self.play(
            FadeIn(dot_a, dot_b),
            Write(label_a),
            Write(label_b)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Draw a straight 'Secant Line' in #E74C3C connecting points A and B.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_SECANT)
        
        secant_line = Line(point_a_coords, point_b_coords, color=COLOR_SECANT)
        
        # Adding slope math for clarity as per outline example
        # Slope = (9-1)/(3-1) = 4
        slope_math = MathTex(
            "m = \\frac{9-1}{3-1} = 4",
            font_size=24,
            color=COLOR_SECANT
        )
        # Fix for Issue 25: use place_in_area for slope_math
        self.place_in_area(slope_math, 'F2', 'F5', scale_factor=0.6)

        self.play(Create(secant_line))
        self.play(Write(slope_math))
        self.wait(2)
        
        # Reset color
        self.lecture[2].set_color(WHITE)
        self.wait(2)
