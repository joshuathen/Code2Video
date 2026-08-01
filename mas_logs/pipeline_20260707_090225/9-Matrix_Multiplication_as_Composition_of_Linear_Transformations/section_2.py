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

class Section2Scene(TeachingScene):
    def construct(self):
        # 1. Setup Layout
        title_text = "Step 1: The First Transformation (Rotation)"
        lecture_lines = [
            "Matrix A acts as a ninety-degree rotation.",
            "[Asset: Robo-Cat] rotates from upright to his side.",
            "Vector v becomes v-prime after this transformation."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Highlight lecture line 1 with its corresponding color
        self.play(self.lecture[0].animate.set_color("#4488FF"))
        
        # Display blue (#4488FF) rotation matrix A [[0, -1], [1, 0]] on screen.
        # Matrix at E4 as per Issue 38
        a_matrix_tex = Text("A = [[0, -1], [1, 0]]", font_size=20, color="#4488FF")
        self.place_at_grid(a_matrix_tex, "E4", scale_factor=0.9)
        self.play(Write(a_matrix_tex))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight lecture line 2
        self.play(self.lecture[1].animate.set_color(WHITE))
        
        # Show Robo-Cat [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/cat.png] 
        # upright on a standard grid (#333333).
        grid_lines = NumberPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=3.5,
            y_length=3.5,
            background_line_style={"stroke_color": "#333333", "stroke_opacity": 1.0}
        )
        # Position grid relative to the area
        self.place_in_area(grid_lines, 'B4', 'D5', scale_factor=1.0)
        
        robo_cat = ImageMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/cat.png")
        # Issue 37: self.place_in_area(robo_cat, 'B4', 'D5', scale_factor=0.6)
        self.place_in_area(robo_cat, 'B4', 'D5', scale_factor=0.6)
        
        self.play(Create(grid_lines), FadeIn(robo_cat))
        self.wait(0.5)

        # Rotate the grid and Robo-Cat 90 degrees counter-clockwise smoothly.
        self.play(
            Rotate(grid_lines, angle=PI/2, about_point=grid_lines.get_center()),
            Rotate(robo_cat, angle=PI/2, about_point=grid_lines.get_center()),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight lecture line 3
        self.play(self.lecture[2].animate.set_color(YELLOW))
        
        # Vector v becomes v-prime after this transformation.
        # Vector label v' at B3 as per Issue 39
        v_prime_tex = Text("v'", font_size=24, color=YELLOW, slant=ITALIC)
        self.place_at_grid(v_prime_tex, "B3", scale_factor=0.8)
        
        # Vector visualization for v'
        v_prime_arrow = Arrow(
            start=grid_lines.get_center(),
            end=grid_lines.get_center() + UP * 1.0,
            color=YELLOW,
            buff=0
        )
        
        self.play(GrowArrow(v_prime_arrow), Write(v_prime_tex))
        self.wait(2)
