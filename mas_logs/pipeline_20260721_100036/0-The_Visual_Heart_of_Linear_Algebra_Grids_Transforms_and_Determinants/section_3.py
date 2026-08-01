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

class Section3Scene(TeachingScene):
    def construct(self):
        # Data from storyboard
        title_text = "Matrices as Morphing Machines"
        lecture_lines = [
            "Matrices act as machines that transform the entire grid.",
            "We only need to track where i and j land.",
            "Linear transformations keep grid lines parallel and evenly spaced.",
            "Watch this matrix rotate the plane ninety degrees counter-clockwise.",
            "The landing spots of i and j define the transformation."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        HIGHLIGHT_COLOR = "#FFD700"
        I_COLOR = "#FF0000"
        J_COLOR = "#00FF00"
        DEG_COLOR = "#FFFFFF"

        # Initialize Grid
        grid = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            background_line_style={"stroke_opacity": 0.4},
            axis_config={"stroke_opacity": 0.6}
        )
        # Using fix from Issue 22: scale_factor=0.55 and area B2-F6 to avoid title and lecture notes.
        self.place_in_area(grid, 'B2', 'F6', scale_factor=0.55)
        origin_point = grid.get_center()

        # === Animation for Lecture Line 1 ===
        # Highlight lecture line 1 (#FFD700). Show i-hat (#FF0000) and j-hat (#00FF00) on a grid.
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT_COLOR))
        
        # Arrows scale relative to the grid's coordinate system
        i_hat = Arrow(origin_point, grid.c2p(1, 0), buff=0, color=I_COLOR, stroke_width=4)
        j_hat = Arrow(origin_point, grid.c2p(0, 1), buff=0, color=J_COLOR, stroke_width=4)
        
        i_label = MathTex("\\hat{i}", color=I_COLOR, font_size=20)
        j_label = MathTex("\\hat{j}", color=J_COLOR, font_size=20)
        
        # Updaters for labels to follow arrow ends
        i_label.add_updater(lambda m: m.next_to(i_hat.get_end(), RIGHT, buff=0.1))
        j_label.add_updater(lambda m: m.next_to(j_hat.get_end(), UP, buff=0.1))

        self.play(
            Create(grid),
            GrowArrow(i_hat),
            GrowArrow(j_hat),
            FadeIn(i_label),
            FadeIn(j_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight lecture line 2 (#FFD700). Transform the grid: i-hat moves to (0,1) and j-hat moves to (-1,0).
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        # Rotation targets (screen coordinates before transformation)
        p_01 = grid.c2p(0, 1)
        p_m10 = grid.c2p(-1, 0)
        
        # Rotation matrix [[0, -1], [1, 0]]
        matrix = [[0, -1], [1, 0]]
        
        self.play(
            grid.animate.apply_matrix(matrix),
            i_hat.animate.put_start_and_end_on(origin_point, p_01),
            j_hat.animate.put_start_and_end_on(origin_point, p_m10),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight lecture line 3 (#FFD700). Draw parallel grid lines that remain straight and evenly spaced.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(HIGHLIGHT_COLOR)
        )
        # Visual emphasis on grid regularity
        self.play(grid.animate.set_stroke(opacity=1), run_time=0.5)
        self.play(grid.animate.set_stroke(opacity=0.4), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Highlight lecture line 4 (#FFD700). Animate labels for i-hat and j-hat following their new positions.
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(HIGHLIGHT_COLOR)
        )
        # Wait to let the viewer see the rotation result and labels.
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Highlight lecture line 5 (#FFD700). Rotate a "90 deg" symbol (#FFFFFF) around the origin.
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        arc = Arc(radius=0.3, start_angle=0, angle=PI/2, color=DEG_COLOR, arc_center=origin_point)
        deg_label = MathTex("90^\\circ", color=DEG_COLOR, font_size=20)
        deg_label.next_to(arc, UR, buff=0.1)
        
        self.play(
            Create(arc),
            FadeIn(deg_label)
        )
        self.wait(2)
