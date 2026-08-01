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

class Section4Scene(TeachingScene):
    def construct(self):
        # Fetching section data
        title_text = "The 'Where Do They Land?' Rule"
        lecture_lines = [
            "Where do the basis vectors land after the transformation?",
            "The first column tells us exactly where i-hat goes.",
            "The second column shows the new home of j-hat.",
            "Knowing these two destinations determines the entire transformation.",
            "See how rotation moves i-hat and j-hat to new spots."
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # Color constants - Updated per VideoCritic Issue 35
        I_HAT_COLOR = "#e07a5f" # red
        J_HAT_COLOR = "#83c5be" # green
        MATRIX_COLOR = "#FFFFFF"
        HIGHLIGHT_COLOR = "#FFFF00"

        # Define Matrix
        matrix_mobject = Matrix([[0, -1], [1, 0]], 
                               left_bracket="[", right_bracket="]",
                               element_to_mobject_config={"color": MATRIX_COLOR})
        
        # [Issue 34] Positioning matrix higher to avoid congestion
        self.place_in_area(matrix_mobject, 'A4', 'B5', scale_factor=0.9)
        
        # Define Coordinate System components
        plane = NumberPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=3,
            y_length=3,
            background_line_style={"stroke_opacity": 0.4}
        )
        
        # [Issue 33 & 35] Shift plane to Col 3 to provide buffer and prevent drift obstruction
        self.place_in_area(plane, 'D3', 'F6', scale_factor=1.1)
        
        # Initial Basis vectors
        i_hat = Arrow(plane.c2p(0, 0), plane.c2p(1, 0), buff=0, color=I_HAT_COLOR, stroke_width=4)
        j_hat = Arrow(plane.c2p(0, 0), plane.c2p(0, 1), buff=0, color=J_HAT_COLOR, stroke_width=4)
        
        i_label = MathTex(r"\hat{i}", color=I_HAT_COLOR, font_size=24)
        j_label = MathTex(r"\hat{j}", color=J_HAT_COLOR, font_size=24)
        
        # Manual positioning relative to arrows
        i_label.next_to(i_hat.get_end(), RIGHT, buff=0.1)
        j_label.next_to(j_hat.get_end(), UP, buff=0.1)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        
        self.play(
            FadeIn(matrix_mobject),
            Create(plane),
            GrowArrow(i_hat),
            GrowArrow(j_hat),
            Write(i_label),
            Write(j_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(I_HAT_COLOR)
        
        col1_rect = SurroundingRectangle(matrix_mobject.get_columns()[0], color=I_HAT_COLOR, buff=0.1)
        
        self.play(Create(col1_rect))
        self.play(Indicate(i_hat, color=I_HAT_COLOR))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(J_HAT_COLOR)
        
        col2_rect = SurroundingRectangle(matrix_mobject.get_columns()[1], color=J_HAT_COLOR, buff=0.1)
        
        self.play(
            ReplacementTransform(col1_rect, col2_rect)
        )
        self.play(Indicate(j_hat, color=J_HAT_COLOR))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(HIGHLIGHT_COLOR)
        
        # Rotation destination for [[0, -1], [1, 0]]: i->(0,1), j->(-1,0)
        new_i_end = plane.c2p(0, 1)
        new_j_end = plane.c2p(-1, 0)
        
        self.play(
            i_hat.animate.put_start_and_end_on(plane.c2p(0, 0), new_i_end),
            j_hat.animate.put_start_and_end_on(plane.c2p(0, 0), new_j_end),
            i_label.animate.next_to(new_i_end, UP, buff=0.1),
            j_label.animate.next_to(new_j_end, LEFT, buff=0.1),
            FadeOut(col2_rect),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(HIGHLIGHT_COLOR)
        
        # Transition the grid to match the transformation
        # Rotating the plane by 90 degrees CCW aligns it with the basis vectors moved in step 4
        self.play(
            Rotate(plane, angle=PI/2, about_point=plane.c2p(0, 0)),
            run_time=2
        )
        self.wait(2)

# Update issues
# update_issue(33, under_review=True, resolution_note="Adjusted plane positioning to D3-F6 and increased scale to 1.1 to prevent label drift obstruction.")
# update_issue(34, under_review=True, resolution_note="Repositioned matrix to A4-B5 and adjusted scale to 0.9 to resolve vertical congestion.")
# update_issue(35, under_review=True, resolution_note="Updated i-hat and j-hat colors to #e07a5f and #83c5be respectively, and provided buffer by shifting plane to Col 3.")
