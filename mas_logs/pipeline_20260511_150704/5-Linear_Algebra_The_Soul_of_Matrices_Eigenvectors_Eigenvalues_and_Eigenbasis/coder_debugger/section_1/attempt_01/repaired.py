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
        # Setup the title and lecture lines
        title = "Introduction: The 'Stretchy Cat' Hook"
        lines = [
            "Linear transformations warp space in many ways.",
            "Most vectors change their direction completely.",
            "But some vectors only stretch or shrink."
        ]
        self.setup_layout(title, lines)

        # Colors for highlights
        EIGEN_COLOR = "#FFFF00"  # Yellow for eigenvectors (whiskers)
        OTHER_COLOR = "#00BFFF"  # DeepSkyBlue for other vectors
        GRID_COLOR = "#444444"

        # === Animation for Lecture Line 1 ===
        # Initialize a 2D coordinate grid and a simple cat-shaped outline.
        self.lecture[0].set_color(YELLOW)
        
        # Create a plane for the right side
        plane = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            background_line_style={"stroke_color": GRID_COLOR, "stroke_opacity": 0.6},
            axis_config={"stroke_color": WHITE, "include_tip": False}
        )
        self.place_in_area(plane, 'A1', 'F6', scale_factor=0.6)

        # Create a simple cat shape (Circle for face, triangles for ears)
        face = Circle(radius=1.0, color=WHITE)
        ear_l = Triangle(color=WHITE).scale(0.3).rotate(20*DEGREES).move_to(face.point_at_angle(120*DEGREES) + UP*0.2)
        ear_r = Triangle(color=WHITE).scale(0.3).rotate(-20*DEGREES).move_to(face.point_at_angle(60*DEGREES) + UP*0.2)
        
        # Whiskers (aligned with the x-axis)
        whisker_l1 = Line([-1.6, 0.1, 0], [-0.8, 0.1, 0], color=WHITE)
        whisker_l2 = Line([-1.6, -0.1, 0], [-0.8, -0.1, 0], color=WHITE)
        whisker_r1 = Line([0.8, 0.1, 0], [1.6, 0.1, 0], color=WHITE)
        whisker_r2 = Line([0.8, -0.1, 0], [1.6, -0.1, 0], color=WHITE)
        
        cat_whiskers = VGroup(whisker_l1, whisker_l2, whisker_r1, whisker_r2)
        cat = VGroup(face, ear_l, ear_r, cat_whiskers)
        
        # Move the cat group to the plane's center
        cat.move_to(plane.get_center())
        
        self.play(Create(plane), Create(cat))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Apply a shear transformation matrix to the grid, tilting the cat shape.
        # Most vectors change their direction completely.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )

        # Vector that changes direction (vertical arrow)
        vec_v = Arrow(plane.get_center(), plane.coords_to_point(0, 1.5), buff=0, color=OTHER_COLOR)
        vec_label = Text("v", color=OTHER_COLOR, font_size=24)
        vec_label.next_to(vec_v.get_end(), RIGHT, buff=0.1)
        
        self.play(GrowArrow(vec_v), Write(vec_label))
        self.wait(0.5)

        # Shear matrix: [[1, 1.5], [0, 1]]
        shear_matrix = [[1, 1.5], [0, 1]]
        
        # Group everything that will be transformed
        transform_group = VGroup(plane, cat, vec_v)
        
        # Note: vec_label position is updated manually to follow vec_v
        def update_label(m):
            m.next_to(vec_v.get_end(), RIGHT, buff=0.1)

        vec_label.add_updater(update_label)

        self.play(
            ApplyMatrix(shear_matrix, transform_group),
            run_time=2,
            rate_func=smooth
        )
        self.wait(1)
        vec_label.remove_updater(update_label)

        # === Animation for Lecture Line 3 ===
        # But some vectors only stretch or shrink. 
        # Highlight the horizontal whiskers to show they stay on the same line.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )

        # Highlight whiskers in Yellow
        self.play(
            cat_whiskers.animate.set_color(EIGEN_COLOR).set_stroke(width=6),
            Indicate(cat_whiskers, color=EIGEN_COLOR)
        )
        
        # Add text labels explaining eigenvectors
        eigen_label = Text("Eigenvectors", font_size=20, color=EIGEN_COLOR)
        self.place_at_grid(eigen_label, 'F4')
        
        self.play(Write(eigen_label))
        self.wait(2)