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
        # Setup Title and Lecture Lines based on Stage-3 prompt
        self.setup_layout(
            "The Calculation: The 2x2 Formula", 
            [
                'A 2x2 matrix defines the transformation’s movement.', 
                'The formula ad minus bc computes the scaling factor.', 
                'Calculating our example results in a determinant of six.'
            ]
        )

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FFFF")
        
        # Matrix creation
        m_a = Text("a", font_size=36)
        m_b = Text("b", font_size=36)
        m_c = Text("c", font_size=36)
        m_d = Text("d", font_size=36)
        lb = Text("[", font_size=60)
        rb = Text("]", font_size=60)
        
        # [ a b ]
        # [ c d ]
        row1 = VGroup(m_a, m_b).arrange(RIGHT, buff=0.8)
        row2 = VGroup(m_c, m_d).arrange(RIGHT, buff=0.8)
        matrix_vals = VGroup(row1, row2).arrange(DOWN, buff=0.5)
        
        matrix_obj = VGroup(lb, matrix_vals, rb).arrange(RIGHT, buff=0.2)
        self.place_in_area(matrix_obj, "B1", "C3", scale_factor=0.8)
        
        # Geometric Parallelogram representing the transformation
        # Fix Issue 43: Precise positioning in B4-D6
        origin = self.grid["D4"]
        v1 = np.array([1.2, 0, 0])
        v2 = np.array([0.4, 0.8, 0])
        p_verts = [origin, origin + v1, origin + v1 + v2, origin + v2]
        para_shape = Polygon(*p_verts, color="#00FFFF", fill_opacity=0.3)
        self.place_in_area(para_shape, "B4", "D6", scale_factor=0.7)
        
        self.play(FadeIn(matrix_obj), Create(para_shape))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFF00")
        
        # Formula: ad - bc
        f_a = Text("a", color="#FFFF00", font_size=40)
        f_d = Text("d", color="#FFFF00", font_size=40)
        f_minus = Text("-", color=WHITE, font_size=40)
        f_b = Text("b", color="#FF00FF", font_size=40)
        f_c = Text("c", color="#FF00FF", font_size=40)
        formula = VGroup(f_a, f_d, f_minus, f_b, f_c).arrange(RIGHT, buff=0.15)
        
        # Fix Issue 42: Positioning at D1-E3
        self.place_in_area(formula, "D1", "E3", scale_factor=0.9)
        
        # Highlight components in matrix to match formula
        self.play(
            Write(formula),
            m_a.animate.set_color("#FFFF00"),
            m_d.animate.set_color("#FFFF00"),
            m_b.animate.set_color("#FF00FF"),
            m_c.animate.set_color("#FF00FF")
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FF00")
        
        # Example values: a=3, b=1, c=0, d=2
        ex_a, ex_b, ex_c, ex_d = "3", "1", "0", "2"
        t_a = Text(ex_a, font_size=36, color="#00FF00")
        t_b = Text(ex_b, font_size=36, color="#00FF00")
        t_c = Text(ex_c, font_size=36, color="#00FF00")
        t_d = Text(ex_d, font_size=36, color="#00FF00")
        
        # Numerical Formula: (3)(2) - (1)(0) = 6
        nf_a = Text(ex_a, font_size=40, color="#FFFF00")
        nf_d = Text(ex_d, font_size=40, color="#FFFF00")
        nf_minus = Text("-", font_size=40, color=WHITE)
        nf_b = Text(ex_b, font_size=40, color="#FF00FF")
        nf_c = Text(ex_c, font_size=40, color="#FF00FF")
        nf_res = Text("= 6", font_size=44, color="#00FF00")
        
        num_formula = VGroup(
            Text("(", font_size=40), nf_a, Text(")(", font_size=40), nf_d, Text(")", font_size=40),
            nf_minus,
            Text("(", font_size=40), nf_b, Text(")(", font_size=40), nf_c, Text(")", font_size=40),
            nf_res
        ).arrange(RIGHT, buff=0.1)
        
        # Fix Issue 41: Positioning at E1-F6 with scale 0.8
        self.place_in_area(num_formula, "E1", "F6", scale_factor=0.8)

        # Updated Parallelogram for [3,0] and [1,2]
        origin_ex = self.grid["D4"]
        ev1 = np.array([1.5, 0, 0])
        ev2 = np.array([0.5, 1.0, 0])
        ex_p_verts = [origin_ex, origin_ex + ev1, origin_ex + ev1 + ev2, origin_ex + ev2]
        ex_para = Polygon(*ex_p_verts, color="#00FF00", fill_opacity=0.4)
        ex_label = Text("Area = 6", font_size=24, color="#00FF00")
        ex_label.next_to(ex_para, UP, buff=0.2)
        
        self.play(
            Transform(m_a, t_a),
            Transform(m_b, t_b),
            Transform(m_c, t_c),
            Transform(m_d, t_d),
            FadeIn(num_formula),
            ReplacementTransform(para_shape, ex_para),
            Write(ex_label)
        )
        self.wait(2)
