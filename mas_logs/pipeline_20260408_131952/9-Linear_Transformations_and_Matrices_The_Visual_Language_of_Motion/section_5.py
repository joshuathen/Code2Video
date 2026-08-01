from manim import *
import numpy as np

# Replacement classes to avoid LaTeX dependency (FileNotFoundError: 'latex')
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

class CustomMathText(Text):
    def __init__(self, tex_string, **kwargs):
        # Basic cleanup of LaTeX commands for Text compatibility
        clean_string = (
            tex_string.replace(r"\cdot", "·")
            .replace(r"\hat{i}", "i-hat")
            .replace(r"\hat{j}", "j-hat")
            .replace(r"_{new}", " new")
            .replace(r"\\", "\n")
            .replace(r"\vec{v}", "v")
        )
        super().__init__(clean_string, **kwargs)

class CustomMatrix(VGroup):
    def __init__(self, matrix_data, **kwargs):
        super().__init__()
        element_to_mobject = kwargs.pop("element_to_mobject", Text)
        left_bracket_str = kwargs.pop("left_bracket", "[")
        right_bracket_str = kwargs.pop("right_bracket", "]")
        
        # Build the elements
        rows = []
        for r in matrix_data:
            row_mobs = VGroup(*[element_to_mobject(str(item), font_size=24) for item in r]).arrange(RIGHT, buff=0.6)
            rows.append(row_mobs)
        
        self.elements_group = VGroup(*rows).arrange(DOWN, buff=0.4)
        
        # Create brackets using Text mobjects
        l_bracket = Text(left_bracket_str, font_size=40)
        l_bracket.stretch_to_fit_height(self.elements_group.height + 0.3)
        l_bracket.next_to(self.elements_group, LEFT, buff=0.15)
        
        r_bracket = Text(right_bracket_str, font_size=40)
        r_bracket.stretch_to_fit_height(self.elements_group.height + 0.3)
        r_bracket.next_to(self.elements_group, RIGHT, buff=0.15)
        
        self.add(l_bracket, r_bracket, self.elements_group)

class Section5Scene(TeachingScene):
    def construct(self):
        # Setup title and lecture lines
        lecture_lines = [
            'Matrix multiplication is a recipe for transforming any point.',
            'Highlight the first column and the original x-coordinate.',
            'Scale the new i-hat by his original x-coordinate.',
            'Add the new j-hat scaled by his original y-coordinate.',
            'His nose lands precisely at its new calculated location.'
        ]
        self.setup_layout("The Calculation: Transforming Pixel", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Use Issue 43 Fix: Matrix A in A1-B3
        matrix_a = CustomMatrix([[2, -1], [1, 1]])
        self.place_in_area(matrix_a, 'A1', 'B3', scale_factor=0.6)
        
        # Pixel's nose vector v = [1, 1]
        v_nose = CustomMatrix([[1], [1]])
        self.place_in_area(v_nose, 'A5', 'B5', scale_factor=0.6)
        
        times_sign = Text("×", font_size=30)
        self.place_at_grid(times_sign, 'A4')
        
        self.lecture[0].set_color(WHITE)
        self.play(FadeIn(matrix_a), FadeIn(v_nose), FadeIn(times_sign))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(GREEN)
        # Highlight first column [2, 1] and nose's original x-coordinate (1)
        col1_group = VGroup(matrix_a.elements_group[0][0], matrix_a.elements_group[1][0])
        x_val = v_nose.elements_group[0][0]
        
        rect_col1 = SurroundingRectangle(col1_group, color=GREEN, buff=0.1)
        rect_x = SurroundingRectangle(x_val, color=GREEN, buff=0.1)
        
        self.play(Create(rect_col1), Create(rect_x))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(GREEN)
        # Show transformed i-hat vector [2, 1] being scaled by x-coordinate
        comp1 = CustomMatrix([[2], [1]])
        self.place_in_area(comp1, 'D1', 'E1', scale_factor=0.6)
        
        # Issue 44 Fix: i-hat label at E5
        i_hat_label = CustomMathText(r"\hat{i}", color=GREEN)
        self.place_at_grid(i_hat_label, 'E5', scale_factor=0.8)
        
        self.play(FadeIn(comp1), FadeIn(i_hat_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(RED)
        # Scale transformed j-hat [-1, 1] by y-coordinate (1) and show sum
        col2_group = VGroup(matrix_a.elements_group[0][1], matrix_a.elements_group[1][1])
        y_val = v_nose.elements_group[1][0]
        
        rect_col2 = SurroundingRectangle(col2_group, color=RED, buff=0.1)
        rect_y = SurroundingRectangle(y_val, color=RED, buff=0.1)
        
        plus_sign = Text("+", font_size=30)
        self.place_at_grid(plus_sign, 'D2')
        
        comp2 = CustomMatrix([[-1], [1]])
        self.place_in_area(comp2, 'D3', 'E3', scale_factor=0.6)
        
        # Issue 45 Fix: j-hat label at B4
        j_hat_label = CustomMathText(r"\hat{j}", color=RED)
        self.place_at_grid(j_hat_label, 'B4', scale_factor=0.8)
        
        self.play(Create(rect_col2), Create(rect_y), FadeIn(plus_sign), FadeIn(comp2), FadeIn(j_hat_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(BLUE)
        # Show result [1, 2] and move nose dot to resulting position
        equals_sign = Text("=", font_size=30)
        self.place_at_grid(equals_sign, 'D4')
        
        result_v = CustomMatrix([[1], [2]])
        self.place_in_area(result_v, 'D5', 'E6', scale_factor=0.6)
        
        # Small coordinate grid to visualize the move
        coord_plane = NumberPlane(
            x_range=[-1, 4, 1], y_range=[-1, 4, 1], 
            x_length=2.5, y_length=2.5,
            background_line_style={"stroke_opacity": 0.2}
        )
        self.place_in_area(coord_plane, 'F1', 'F6', scale_factor=1.0)
        
        nose_dot = Dot(color=BLUE)
        nose_dot.move_to(coord_plane.c2p(1, 1))
        
        self.play(
            FadeIn(equals_sign), 
            FadeIn(result_v),
            FadeIn(coord_plane),
            FadeIn(nose_dot)
        )
        self.wait(0.5)
        
        # Calculation: 1*[2, 1] + 1*[-1, 1] = [1, 2]
        self.play(
            nose_dot.animate.move_to(coord_plane.c2p(1, 2)),
            run_time=2
        )
        self.wait(3)
