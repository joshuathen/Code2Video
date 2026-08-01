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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup layout
        lecture_lines = [
            'The cross product links geometry, algebra, and transformations.',
            'It measures area, solves determinants, and maps space.',
            'This unique duality defines its role in 3D.',
            'It remains a fundamental tool in physics and math.',
            'Visualizing these connections clarifies linear algebra.'
        ]
        self.setup_layout("Summary and Synthesis", lecture_lines)

        # Colors
        GEO_COLOR = YELLOW_B
        ALG_COLOR = GREEN_B
        PHYS_COLOR = RED_B
        HIGHLIGHT_COLOR = "#3357FF"

        # === Animation for Lecture Line 1 ===
        # Create the three components representing the three views
        
        # 1. Geometry View: Parallelogram
        v1 = Arrow(ORIGIN, RIGHT * 1.5, buff=0, color=GEO_COLOR)
        v2 = Arrow(ORIGIN, UP * 1.0 + RIGHT * 0.5, buff=0, color=GEO_COLOR)
        poly = Polygon(ORIGIN, RIGHT * 1.5, RIGHT * 2.0 + UP * 1.0, UP * 1.0 + RIGHT * 0.5, 
                       fill_opacity=0.3, fill_color=GEO_COLOR, stroke_width=0)
        geo_group = VGroup(poly, v1, v2)
        # Fix Issue 42: Move to A3-B6 and increase scale
        self.place_in_area(geo_group, 'A3', 'B6', scale_factor=0.7)
        
        # 2. Algebra View: Matrix Determinant
        # Manual construction to avoid LaTeX dependency (Matrix class uses MathTex for brackets)
        m_data = [["i", "j", "k"], ["v_x", "v_y", "v_z"], ["w_x", "w_y", "w_z"]]
        matrix_cols = []
        for j in range(3):
            col = VGroup(*[Text(m_data[i][j], color=ALG_COLOR, font_size=30) for i in range(3)]).arrange(DOWN, buff=0.5)
            matrix_cols.append(col)
        matrix_elements = VGroup(*matrix_cols).arrange(RIGHT, buff=0.7)
        l_bar = Line(UP, DOWN, color=WHITE).stretch_to_fit_height(matrix_elements.height + 0.2).next_to(matrix_elements, LEFT, buff=0.2)
        r_bar = Line(UP, DOWN, color=WHITE).stretch_to_fit_height(matrix_elements.height + 0.2).next_to(matrix_elements, RIGHT, buff=0.2)
        matrix_obj = VGroup(matrix_elements, l_bar, r_bar)
        # Fix Issue 41: Move to C3-D6 and increase scale
        self.place_in_area(matrix_obj, 'C3', 'D6', scale_factor=0.7)

        # 3. Transformation/Physics View: Torque/Wrench
        # A simple lever arm and force vector
        lever = Line(LEFT * 0.8, RIGHT * 0.8, stroke_width=8, color=GRAY_B)
        pivot = Dot(LEFT * 0.8, color=WHITE)
        force = Arrow(RIGHT * 0.8, RIGHT * 0.8 + UP * 1.2, buff=0, color=PHYS_COLOR)
        torque_arc = Arc(radius=0.4, start_angle=0, angle=PI/2, arc_center=LEFT * 0.8, color=PHYS_COLOR).add_tip()
        wrench_group = VGroup(lever, pivot, force, torque_arc)
        # Fix Issue 43: Move to E3-F6 and increase scale
        self.place_in_area(wrench_group, 'E3', 'F6', scale_factor=0.7)

        # Initial display
        self.play(
            FadeIn(geo_group),
            FadeIn(matrix_obj),
            FadeIn(wrench_group),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight sequence
        self.play(self.lecture[1].animate.set_color(HIGHLIGHT_COLOR))
        
        # Flash Geometric view
        box1 = SurroundingRectangle(geo_group, color=HIGHLIGHT_COLOR, buff=0.2)
        self.play(Create(box1))
        self.play(FadeOut(box1), run_time=0.5)
        
        # Flash Algebraic view
        box2 = SurroundingRectangle(matrix_obj, color=HIGHLIGHT_COLOR, buff=0.2)
        self.play(Create(box2))
        self.play(FadeOut(box2), run_time=0.5)
        
        # Flash Physics view
        box3 = SurroundingRectangle(wrench_group, color=HIGHLIGHT_COLOR, buff=0.2)
        self.play(Create(box3))
        self.play(FadeOut(box3), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight logic connection
        self.play(self.lecture[2].animate.set_color(GEO_COLOR))
        self.play(geo_group.animate.scale(1.1).set_color(WHITE), run_time=0.8)
        self.play(geo_group.animate.scale(1/1.1).set_color(GEO_COLOR), run_time=0.8)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Focus on the transformation/physics
        self.play(self.lecture[3].animate.set_color(PHYS_COLOR))
        # Use pivot.get_center() for accurate rotation
        pivot_center = pivot.get_center()
        self.play(wrench_group.animate.rotate(PI/6, about_point=pivot_center), run_time=1)
        self.play(wrench_group.animate.rotate(-PI/6, about_point=pivot_center), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Fade out and show final synthesis title
        self.play(self.lecture[4].animate.set_color(ALG_COLOR))
        self.wait(1)
        
        self.play(
            FadeOut(self.lecture),
            FadeOut(self.title),
            FadeOut(geo_group),
            FadeOut(matrix_obj),
            FadeOut(wrench_group),
            run_time=1.5
        )
        
        final_title = Text("The Duality of Cross Products", font_size=42, color=HIGHLIGHT_COLOR)
        self.add(final_title)
        self.play(Write(final_title))
        self.play(Indicate(final_title, color=WHITE))
        self.wait(2)
