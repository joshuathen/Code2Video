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

class Section1Scene(TeachingScene):
    def construct(self):
        # Setup the layout
        self.setup_layout(
            "Prerequisite: Linear Transformations as Space Warping",
            [
                'Matrix multiplication warps the entire coordinate space.',
                'Watch our Elastic-Cat stretch under this linear transformation.',
                'Most vectors shift away from their original lines.'
            ]
        )

        # Colors
        PLANE_COLOR = "#1890FF"
        VECTOR_COLOR = "#FFFFFF"
        HIGHLIGHT_COLOR = "#FF4D4F"
        
        # Shear matrix definition
        shear_matrix = [[1, 1], [0, 1]]

        # === Animation for Lecture Line 1 ===
        # Matrix multiplication warps the entire coordinate space.
        self.play(self.lecture[0].animate.set_color(PLANE_COLOR))
        
        plane = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            background_line_style={"stroke_color": PLANE_COLOR, "stroke_opacity": 0.4},
            axis_config={"stroke_color": PLANE_COLOR},
        )
        # Fix: Line 12 (Issue 25)
        self.place_in_area(plane, 'B2', 'F6', scale_factor=0.8)
        
        self.play(Create(plane))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Watch our Elastic-Cat stretch under this linear transformation.
        self.play(self.lecture[1].animate.set_color(HIGHLIGHT_COLOR))
        
        # Fix: Line 18 (Issue 26 & 20)
        cat = ImageMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/cat.png")
        self.place_at_grid(cat, 'D4', scale_factor=0.4)
        
        # Define vectors for Line 3 as well, but create them now to transform
        v1 = Arrow(ORIGIN, RIGHT * 1.5, buff=0, color=VECTOR_COLOR, stroke_width=6)
        v2 = Arrow(ORIGIN, UP * 1.5, buff=0, color=VECTOR_COLOR, stroke_width=6)
        v3 = Arrow(ORIGIN, (UP + LEFT) * 1.5, buff=0, color=VECTOR_COLOR, stroke_width=6)
        vector_group = VGroup(v1, v2, v3)
        
        # Fix: Line 24 (Issue 27)
        self.place_in_area(vector_group, 'C3', 'E5', scale_factor=0.6)
        
        # Store original positions for spans in Line 3
        center_point = plane.get_center()
        spans = VGroup(*[
            DashedLine(
                center_point - (v.get_end() - center_point) * 2,
                center_point + (v.get_end() - center_point) * 2,
                color=WHITE,
                stroke_opacity=0.3
            ) for v in vector_group
        ])

        self.play(FadeIn(cat), Create(vector_group))
        self.wait(1)

        # Transformation Animation
        self.play(
            plane.animate.apply_matrix(shear_matrix, about_point=center_point),
            cat.animate.apply_matrix(shear_matrix, about_point=center_point),
            vector_group.animate.apply_matrix(shear_matrix, about_point=center_point),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Most vectors shift away from their original lines.
        self.play(self.lecture[2].animate.set_color(HIGHLIGHT_COLOR))
        
        # Show the original spans to demonstrate shift
        self.play(FadeIn(spans))
        
        # Highlight vectors that moved off span (Vector 2 and 3)
        # Vector 1 (Right) is an eigenvector for this shear, so it stays on its span.
        self.play(
            vector_group[1].animate.set_color(HIGHLIGHT_COLOR),
            vector_group[2].animate.set_color(HIGHLIGHT_COLOR),
        )
        
        self.play(
            vector_group[1].animate.scale(1.2, about_point=center_point),
            vector_group[2].animate.scale(1.2, about_point=center_point),
        )
        self.play(
            vector_group[1].animate.scale(1/1.2, about_point=center_point),
            vector_group[2].animate.scale(1/1.2, about_point=center_point),
        )
        
        self.wait(2)
