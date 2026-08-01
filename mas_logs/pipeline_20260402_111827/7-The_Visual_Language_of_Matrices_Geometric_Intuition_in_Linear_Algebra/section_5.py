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

class Section5Scene(TeachingScene):
    def construct(self):
        # Setup content and layout
        title_text = "The Determinant: The Scaling Factor"
        lecture_lines = [
            "The determinant measures how much areas change.",
            "This unit rug stretches as the grid warps.",
            "Its new area represents the matrix determinant.",
            "A determinant of three means the area tripled.",
            "If it is zero, the area completely collapses."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors for matching lecture lines
        COLOR_1 = "#FFFF00" # Yellow
        COLOR_2 = "#00FFFF" # Cyan
        COLOR_3 = "#FF5555" # Light Red
        COLOR_4 = "#FFA500" # Orange
        COLOR_5 = "#FFFFFF" # White

        # Setup Axes - Persistent mobject in area B2-E5
        axes = NumberPlane(
            x_range=[-1, 4, 1],
            y_range=[-1, 2, 1],
            x_length=4,
            y_length=2.5,
            background_line_style={"stroke_color": BLUE_E, "stroke_width": 1, "stroke_opacity": 0.5}
        ).set_z_index(0)
        self.place_in_area(axes, "B2", "E5")

        # Setup Rug Asset [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/rug.svg]
        rug_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/rug.svg"
        rug = SVGMobject(rug_path)
        rug.set_color(COLOR_1)
        
        # Scale rug to 1x1 based on axes units
        x_unit = axes.c2p(1,0)[0] - axes.c2p(0,0)[0]
        y_unit = axes.c2p(0,1)[1] - axes.c2p(0,0)[1]
        rug.stretch_to_fit_width(x_unit)
        rug.stretch_to_fit_height(y_unit)
        # Position so corners are at (0,0) to (1,1)
        rug.move_to(axes.c2p(0.5, 0.5))

        # Persistent Labels
        det_label = Text("Determinant = 1", font_size=22, color=WHITE)
        # Solving Issue 54, 55, 56 by placing at A4 with 0.9 scale
        self.place_at_grid(det_label, "A4", scale_factor=0.9)
        
        area_value = Text("3", font_size=20, color=WHITE)
        area_value.set_opacity(0)
        self.place_at_grid(area_value, "C4") # Initial hidden position

        # Group for transformations
        scene_content = VGroup(axes, rug)

        # === Animation for Lecture Line 1 ===
        # Highlight a 1x1 yellow square #FFFF00 'rug'
        self.play(
            self.lecture[0].animate.set_color(COLOR_1),
            Create(axes),
            FadeIn(rug),
            FadeIn(det_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Animate the grid transformation, stretching the yellow square into a parallelogram.
        # Determinant = 3 state: Matrix [[3, 1], [0, 1]]
        matrix_3 = [[3, 1], [0, 1]]
        
        self.play(
            self.lecture[1].animate.set_color(COLOR_2),
            scene_content.animate.apply_matrix(matrix_3),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Its new area represents the matrix determinant.
        # Reveal the area value inside the rug
        self.play(
            self.lecture[2].animate.set_color(COLOR_3),
            area_value.animate.set_opacity(1).move_to(rug.get_center())
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # A determinant of three means the area tripled.
        # Update text to 'Determinant = 3' [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/rug.svg]
        det_label_new = Text("Determinant = 3", font_size=22, color=WHITE)
        det_label_new.move_to(det_label.get_center())

        self.play(
            self.lecture[3].animate.set_color(COLOR_4),
            rug.animate.set_color(COLOR_4),
            Transform(det_label, det_label_new)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Animate the grid collapsing into a single line, showing the determinant reaching 0.
        # Matrix [[1, 1], [1, 1]] has determinant 0.
        # Calculate relative transform: Target @ Inverse(Current)
        matrix_0 = [[1, 1], [1, 1]]
        m_collapse = np.dot(matrix_0, np.linalg.inv(matrix_3))
        
        det_label_final = Text("Determinant = 0", font_size=22, color=WHITE)
        det_label_final.move_to(det_label.get_center())

        self.play(
            self.lecture[4].animate.set_color(COLOR_5),
            scene_content.animate.apply_matrix(m_collapse),
            Transform(det_label, det_label_final),
            FadeOut(area_value),
            run_time=2
        )
        self.wait(2)
