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

class Section2Scene(TeachingScene):
    def make_sieve(self, row_char, gap_col_char, label_text):
        """Creates a horizontal sieve with a gap at the specified grid column."""
        center_pos = self.grid[f"{row_char}{gap_col_char}"]
        left_end = self.grid[f"{row_char}1"]
        right_end = self.grid[f"{row_char}6"]
        
        # Sieve lines with a gap in the middle
        left_line = Line(left_end, center_pos + LEFT*0.3, color="#ADD8E6")
        right_line = Line(center_pos + RIGHT*0.3, right_end, color="#ADD8E6")
        
        # Label for the power of 2
        label = MathTex(label_text, color="#ADD8E6", font_size=24).next_to(left_line, LEFT, buff=0.1)
        
        return VGroup(left_line, right_line, label)

    def construct(self):
        self.setup_layout("Prerequisite: The Power of Two (p-adic Valuation)", 
                          ["The 2-adic valuation measures the '2-ness' of a number.", 
                           "It is the highest power of 2 dividing an integer.", 
                           "Higher powers fall deeper through our mathematical sieves."])
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#ADD8E6")
        
        # Formula display
        # Issue 22: Move v2_formula from A3 to A4 for better balance
        v2_formula = MathTex("v_2(n)", color="#ADD8E6", font_size=36)
        self.place_at_grid(v2_formula, "A4")
        
        # Stack of sieves (2^1 to 2^4)
        sieves = VGroup()
        exponents = ["2^1", "2^2", "2^3", "2^4"]
        for i, exp in enumerate(exponents):
            row = chr(ord('B') + i)
            sieve = self.make_sieve(row, "3", exp)
            sieves.add(sieve)
            
        self.play(Write(v2_formula))
        self.play(FadeIn(sieves))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#ADD8E6")
        
        # Number 40 falling through
        # Issue 21: To avoid overlap with the formula at A4, we ensure num40 starts at A3.
        # The overlap is resolved because v2_formula was moved to A4.
        num40 = Text("40", color="#ADD8E6", font_size=32)
        self.place_at_grid(num40, "A3")
        
        self.play(FadeIn(num40))
        
        # 40 = 2^3 * 5, so it passes 2^1, 2^2, 2^3 and stops at 2^4
        # Movement: Row B (2^1), Row C (2^2), Row D (2^3), Row E (2^4)
        self.play(num40.animate.move_to(self.grid["B3"]), run_time=0.6)
        self.play(num40.animate.move_to(self.grid["C3"]), run_time=0.6)
        self.play(num40.animate.move_to(self.grid["D3"]), run_time=0.6)
        # Stops at E3 (where 2^4 is)
        self.play(num40.animate.move_to(self.grid["E3"] + UP*0.2), run_time=0.6)
        
        v2_40_val = MathTex("v_2(40) = 3", color="#ADD8E6", font_size=32)
        v2_40_val.next_to(num40, RIGHT, buff=0.5)
        self.play(Write(v2_40_val))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#ADD8E6")
        
        # Number 7 failing at the first sieve
        # Issue 23: Ensure final position for num7 is at B5 (on the sieve).
        num7 = Text("7", color="#ADD8E6", font_size=32)
        self.place_at_grid(num7, "A5") # Start high
        
        self.play(FadeIn(num7))
        # 7 hits the solid part of Sieve 1 (Row B, Col 5)
        self.play(num7.animate.move_to(self.grid["B5"] + UP*0.2), run_time=1)
        
        v2_7_val = MathTex("v_2(7) = 0", color="#ADD8E6", font_size=32)
        v2_7_val.next_to(num7, RIGHT, buff=0.5)
        self.play(Write(v2_7_val))
        self.wait(2)
