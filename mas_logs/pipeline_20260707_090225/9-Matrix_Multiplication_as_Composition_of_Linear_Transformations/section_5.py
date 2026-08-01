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

class Section5Scene(TeachingScene):
    def create_text_matrix(self, matrix_data, color=WHITE):
        """
        Creates a matrix VGroup using Text mobjects to avoid LaTeX overhead.
        """
        matrix_vgroup = VGroup()
        for row_data in matrix_data:
            row_vgroup = VGroup(*[Text(str(item), font_size=24, color=color) for item in row_data])
            row_vgroup.arrange(RIGHT, buff=0.7)
            matrix_vgroup.add(row_vgroup)
        
        matrix_vgroup.arrange(DOWN, buff=0.5)
        
        # Draw brackets
        h = matrix_vgroup.height + 0.2
        w = 0.15
        
        l_bracket = VGroup(
            Line(ORIGIN, [w, 0, 0]),
            Line(ORIGIN, [0, -h, 0]),
            Line([0, -h, 0], [w, -h, 0])
        ).set_color(color)
        
        r_bracket = VGroup(
            Line(ORIGIN, [-w, 0, 0]),
            Line(ORIGIN, [0, -h, 0]),
            Line([0, -h, 0], [-w, -h, 0])
        ).set_color(color)
        
        l_bracket.next_to(matrix_vgroup, LEFT, buff=0.2)
        r_bracket.next_to(matrix_vgroup, RIGHT, buff=0.2)
        
        return VGroup(l_bracket, matrix_vgroup, r_bracket)

    def construct(self):
        # 1. Setup Layout
        lecture_lines = [
            "Matrix B acts on the columns of Matrix A.",
            "We track the final destination of original basis vectors.",
            "This links visual motion to the numerical dot product."
        ]
        self.setup_layout("Mechanics: Why the Dot Product Works", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Highlight: Red for i-hat's landing (first col of A), Orange for Matrix B
        self.play(self.lecture[0].animate.set_color("#FF0000"))
        
        data_a = [["a", "b"], ["c", "d"]]
        data_b = [["e", "f"], ["g", "h"]]
        
        matrix_a = self.create_text_matrix(data_a, color=BLUE)
        matrix_b = self.create_text_matrix(data_b, color="#FF8844") # Orange Matrix B
        
        # Group them for display
        matrix_group = VGroup(matrix_b, matrix_a).arrange(RIGHT, buff=0.5)
        # Issue 46 Fix
        self.place_in_area(matrix_group, 'B2', 'C5', scale_factor=0.8)
        
        self.play(FadeIn(matrix_group))
        
        # Highlight first column of Matrix A (where i-hat landed after A)
        col_1_elements = VGroup(matrix_a[1][0][0], matrix_a[1][1][0])
        highlight_box = SurroundingRectangle(col_1_elements, color="#FF0000", buff=0.1)
        
        self.play(Create(highlight_box), col_1_elements.animate.set_color("#FF0000"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF8844"))
        
        # Show Matrix B acting on the red column vector
        col_vec_a = self.create_text_matrix([["a"], ["c"]], color="#FF0000")
        calc_group = VGroup(matrix_b.copy(), col_vec_a).arrange(RIGHT, buff=0.4)
        self.place_in_area(calc_group, 'D2', 'D5', scale_factor=0.8)
        
        self.play(TransformFromCopy(VGroup(matrix_b, col_1_elements), calc_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(WHITE))
        
        # Result of multiplication goes into column 1 of product matrix C
        data_c = [["ea+fc", "eb+fd"], ["ga+hc", "gb+hd"]]
        matrix_c = self.create_text_matrix(data_c, color=WHITE)
        result_box_group = VGroup(Text("Product Matrix C", font_size=20), matrix_c).arrange(DOWN, buff=0.3)
        # Issue 48 Fix
        self.place_in_area(result_box_group, 'B2', 'D5', scale_factor=0.9)
        
        # Highlight the destination of i-hat in the product matrix
        col_1_c = VGroup(matrix_c[1][0][0], matrix_c[1][1][0])
        col_1_c.set_color("#FF0000")
        
        # Issue 47 Fix
        combined_label = Text("Final basis vector destination", font_size=20, color="#FF8844")
        self.place_in_area(combined_label, 'E2', 'E5', scale_factor=0.6)
        
        self.play(
            FadeOut(matrix_group),
            FadeOut(calc_group),
            FadeOut(highlight_box),
            FadeIn(result_box_group),
            Write(combined_label)
        )
        self.wait(2)
