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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            'Welcome to the coordinate plane, our mathematical playground.', 
            'Meet Pixel, our character positioned at the grid center.', 
            'A vector arrow locates his nose precisely in space.', 
            'Multiple vectors together define his entire visual shape.', 
            'Transforming means moving every point in a systematic way.'
        ]
        self.setup_layout("Introduction: Pixel the Cat and the Grid", lecture_lines)
        
        # Define Colors
        GRID_COLOR = "#888888"
        NOSE_COLOR = "#0000FF"
        VECTOR_COLOR = "#FFFFFF"
        FEATURE_VECTOR_COLOR = "#ADD8E6"
        HIGHLIGHT_COLOR = YELLOW

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        
        # Coordinate grid
        plane = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=5,
            y_length=5,
            background_line_style={"stroke_color": GRID_COLOR, "stroke_opacity": 0.6},
            axis_config={"stroke_color": GRID_COLOR}
        )
        self.place_in_area(plane, "A1", "F6")
        
        self.play(Create(plane))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_COLOR)

        # Pixel the Cat (Simple shape construction)
        head = Circle(radius=0.4, color=WHITE, fill_opacity=0.1)
        l_ear = Triangle(color=WHITE).scale(0.12).rotate(30*DEGREES).shift(LEFT*0.25 + UP*0.35)
        r_ear = Triangle(color=WHITE).scale(0.12).rotate(-30*DEGREES).shift(RIGHT*0.25 + UP*0.35)
        nose = Dot(ORIGIN, color=NOSE_COLOR)
        l_eye = Dot(LEFT*0.15 + UP*0.15, color=WHITE, radius=0.03)
        r_eye = Dot(RIGHT*0.15 + UP*0.15, color=WHITE, radius=0.03)
        
        pixel = VGroup(head, l_ear, r_ear, nose, l_eye, r_eye)
        
        # Fix 31: Place pixel at grid center
        self.place_in_area(pixel, 'C3', 'D4', scale_factor=0.5)
        
        self.play(FadeIn(pixel))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)

        # A vector arrow locates his nose precisely in space.
        nose_vector = Arrow(plane.get_origin(), nose.get_center(), color=VECTOR_COLOR, buff=0)
        
        # Fix 32: Positioning the nose vector
        self.place_at_grid(nose_vector, 'C3', scale_factor=0.7)
        
        self.play(GrowArrow(nose_vector))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(HIGHLIGHT_COLOR)

        # Multiple vectors together define his entire visual shape.
        feature_points = [l_ear.get_top(), r_ear.get_top(), l_eye.get_center(), r_eye.get_center()]
        vector_group = VGroup()
        for pt in feature_points:
            vector_group.add(Arrow(
                plane.get_origin(), pt, 
                color=FEATURE_VECTOR_COLOR, 
                buff=0, 
                stroke_width=2, 
                max_tip_length_to_length_ratio=0.15
            ))
            
        self.play(Create(vector_group))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(HIGHLIGHT_COLOR)

        # Transforming means moving every point in a systematic way.
        matrix = [[1.2, 0.4], [0.2, 0.9]]
        
        # Group character, vectors, and grid to transform together
        all_content = VGroup(plane, pixel, nose_vector, vector_group)
        
        # Perform the transformation
        self.play(
            all_content.animate.apply_matrix(matrix),
            run_time=3
        )
        
        # Fix 33: Line 105 call as suggested by critic
        self.place_in_area(vector_group, 'C3', 'D4', scale_factor=0.5)
        
        self.wait(2)
