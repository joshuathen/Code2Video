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
        self.setup_layout(
            "Prerequisite: Binary Mapping and XOR Magic", 
            [
                'First, we label every square from zero to sixty-three.', 
                'We represent these numbers using six-bit binary codes.', 
                'Now, let’s introduce XOR, our key mathematical tool.', 
                'XOR compares bits: same gives zero, different gives one.', 
                'It acts as a toggle without carrying any numbers.'
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        
        # Create 8x8 chessboard
        squares = VGroup(*[
            Square(side_length=0.5, stroke_width=2, color=WHITE) 
            for _ in range(64)
        ]).arrange_in_grid(8, 8, buff=0)
        
        decimal_labels = VGroup(*[
            Text(str(i), font_size=16, color=WHITE) 
            for i in range(64)
        ])
        
        for i, label in enumerate(decimal_labels):
            label.move_to(squares[i].get_center())
            
        chessboard = VGroup(squares, decimal_labels)
        self.place_in_area(chessboard, 'A1', 'F6', scale_factor=0.9)
        
        self.play(Create(squares), run_time=1.5)
        self.play(LaggedStart(*[Write(lbl) for lbl in decimal_labels], lag_ratio=0.02), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFD700")
        
        binary_labels = VGroup(*[
            Text(format(i, '06b'), font_size=10, color="#FFD700") 
            for i in range(64)
        ])
        
        for i, b_lbl in enumerate(binary_labels):
            b_lbl.move_to(squares[i].get_center())
            
        self.play(Transform(decimal_labels, binary_labels), run_time=2)
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#1E90FF")
        
        xor_text = Text("XOR", font_size=72, color="#1E90FF", weight=BOLD)
        self.place_in_area(xor_text, 'B2', 'E5')
        
        self.play(FadeOut(chessboard), FadeIn(xor_text))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00FF00")
        
        # Truth Table Construction
        def get_xor_row(a, b, res):
            a_str = "1" if a else "0"
            b_str = "1" if b else "0"
            r_str = "1" if res else "0"
            a_col = "#00FF00" if a else "#FF0000"
            b_col = "#00FF00" if b else "#FF0000"
            r_col = "#00FF00" if res else "#FF0000"
            
            row = VGroup(
                Text(a_str, color=a_col),
                Text(" ^ ", color=WHITE),
                Text(b_str, color=b_col),
                Text(" = ", color=WHITE),
                Text(r_str, color=r_col)
            ).arrange(RIGHT, buff=0.2)
            return row

        tt_rows = VGroup(
            get_xor_row(0, 0, 0),
            get_xor_row(1, 1, 0),
            get_xor_row(1, 0, 1),
            get_xor_row(0, 1, 1)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        
        self.place_in_area(tt_rows, 'B2', 'E5', scale_factor=0.8)
        
        self.play(FadeOut(xor_text))
        self.play(LaggedStart(*[Write(row) for row in tt_rows], lag_ratio=0.3), run_time=2)
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(WHITE)
        
        # Vertical Calculation 1101 ^ 1011 = 0110
        val1 = "1101"
        val2 = "1011"
        res  = "0110"
        
        v_top = VGroup(*[Text(d, font_size=36) for d in val1]).arrange(RIGHT, buff=0.4)
        v_mid = VGroup(*[Text(d, font_size=36) for d in val2]).arrange(RIGHT, buff=0.4)
        v_res = VGroup(*[Text(d, font_size=36) for d in res]).arrange(RIGHT, buff=0.4)
        
        op_symbol = Text("^", font_size=36).next_to(v_mid, LEFT, buff=0.5)
        line_sep = Line(LEFT, RIGHT).scale(2).next_to(v_mid, DOWN, buff=0.2)
        v_res.next_to(line_sep, DOWN, buff=0.2)
        
        calc_group = VGroup(v_top, v_mid, op_symbol, line_sep, v_res)
        self.place_in_area(calc_group, 'B2', 'E5', scale_factor=1.0)
        
        self.play(FadeOut(tt_rows))
        self.play(Write(VGroup(v_top, v_mid, op_symbol, line_sep)))
        self.wait(0.5)
        
        # Highlight columns one by one
        for i in range(4):
            rect = SurroundingRectangle(VGroup(v_top[i], v_mid[i]), color=YELLOW, buff=0.1)
            self.play(Create(rect), run_time=0.3)
            self.play(Write(v_res[i]), run_time=0.3)
            self.play(FadeOut(rect), run_time=0.2)
            
        self.wait(3)
