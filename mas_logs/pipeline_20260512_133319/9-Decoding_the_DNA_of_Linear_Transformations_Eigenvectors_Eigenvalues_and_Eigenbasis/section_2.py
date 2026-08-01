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
        # Data
        title_text = "Defining Eigenvectors: The Unchanging Directions"
        lecture_lines = [
            'Some special vectors stay on their original span.',
            'Watch the whisker as the cat stretches diagonally.',
            'It stays horizontal while everything else tilts.',
            'These unchanging directions are called eigenvectors.',
            "They define the transformation's fundamental axes."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        CAT_COLOR = "#FAAD14"
        PLANE_COLOR = "#1890FF"
        EIGEN_COLOR = "#52C41A"
        HIGHLIGHT_COLOR = YELLOW

        # === Animation for Lecture Line 1 ===
        # Line 1: Some special vectors stay on their original span.
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        
        # Display the cat shape [Asset: cat.png] on the NumberPlane
        plane = NumberPlane(
            x_range=[-4, 4, 1], y_range=[-4, 4, 1],
            x_length=5, y_length=5,
            background_line_style={"stroke_color": PLANE_COLOR, "stroke_opacity": 0.4}
        )
        
        cat_img = ImageMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/cat.png")
        cat_img.set_color(CAT_COLOR)
        cat_img.scale_to_fit_width(1.5)
        
        # Group them to place them in the right-side area
        viz_group = Group(plane, cat_img)
        self.place_in_area(viz_group, 'A1', 'F6', scale_factor=1.0)
        
        self.play(FadeIn(plane), FadeIn(cat_img))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line 2: Watch the whisker as the cat stretches diagonally.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_COLOR)
        
        # Define the whisker vector (horizontal, staying on x-axis)
        # It starts from the origin (cat nose) pointing left
        whisker_start = plane.get_center()
        whisker_end = plane.coords_to_point(-1, 0)
        
        whisker_vector = Arrow(
            start=whisker_start,
            end=whisker_end,
            buff=0,
            color=EIGEN_COLOR,
            stroke_width=6
        )
        
        self.play(GrowArrow(whisker_vector))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line 3: It stays horizontal while everything else tilts.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)
        
        # Matrix to apply diagonal stretch/shear: [[2, 1], [0, 1]]
        # In this matrix, the vector [1, 0] stays on the x-axis but scales by 2.
        matrix = [[2, 1], [0, 1]]
        
        # Animate the stretch
        self.play(
            cat_img.animate.apply_matrix(matrix, about_point=plane.get_center()),
            whisker_vector.animate.apply_matrix(matrix, about_point=plane.get_center()),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Line 4: These unchanging directions are called eigenvectors.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(HIGHLIGHT_COLOR)
        
        # Draw span line extending through the whisker
        span_line = DashedLine(
            start=plane.coords_to_point(-4, 0),
            end=plane.coords_to_point(4, 0),
            color=EIGEN_COLOR,
            stroke_width=2,
            dash_length=0.1
        )
        
        self.play(Create(span_line))
        # Ensure whisker is on top
        self.add(whisker_vector)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Line 5: They define the transformation's fundamental axes.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(HIGHLIGHT_COLOR)
        
        # Label at A2 to avoid overlap with cat visualization center (Issue 28)
        label = Text("Eigenvector", font_size=20, color=EIGEN_COLOR)
        self.place_at_grid(label, 'A2', scale_factor=1.0)
        
        pointer = Arrow(
            start=label.get_bottom(),
            end=whisker_vector.get_center(),
            buff=0.1,
            color=EIGEN_COLOR,
            stroke_width=3,
            max_tip_length_to_length_ratio=0.15
        )
        
        self.play(Write(label), GrowArrow(pointer))
        self.wait(2)
