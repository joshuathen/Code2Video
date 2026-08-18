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

class Section6Scene(TeachingScene):
    def construct(self):
        # Initial layout setup
        self.setup_layout("Conclusion: Beyond the Pattern", [
            "A few cases never constitute a mathematical proof.",
            "Always verify patterns with rigorous logic and derivation.",
            "Mathematical truth requires more than just a good start."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Display a table comparing '2^(n-1)' predictions vs 'Actual' values for n=1 to 7.
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Table Header
        h_n = MathTex("n", color=WHITE)
        h_pred = MathTex("2^{n-1}", color=WHITE)
        h_act = MathTex(r"\text{Actual}", color=WHITE)
        header = VGroup(h_n, h_pred, h_act).arrange(RIGHT, buff=0.8)
        
        # Table Values for n=1 to 7
        vals = [
            ("1", "1", "1"),
            ("2", "2", "2"),
            ("3", "4", "4"),
            ("4", "8", "8"),
            ("5", "16", "16"),
            ("6", "32", "31"),
            ("7", "64", "57"),
        ]
        
        rows_list = [header]
        for n_val, p_val, a_val in vals:
            r_n = MathTex(n_val, color=WHITE)
            r_p = MathTex(p_val, color=WHITE)
            r_a = MathTex(a_val, color=WHITE)
            row = VGroup(r_n, r_p, r_a).arrange(RIGHT, buff=0.8)
            # Align each element in row to the header column elements
            for i in range(3):
                row[i].align_to(header[i], LEFT)
            rows_list.append(row)
            
        table = VGroup(*rows_list).arrange(DOWN, buff=0.2)
        
        # Resolve Issue 31: Place table in C1-F6 to leave space for final message
        self.place_in_area(table, 'C1', 'F6', scale_factor=0.7)
        
        self.play(FadeIn(table, shift=UP * 0.2), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Always verify patterns with rigorous logic and derivation.
        # Highlight the rows for n=6 (32 vs 31) and n=7 (64 vs 57) with red text (#FF0000).
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW),
        )
        
        # Rows n=6 and n=7 correspond to indices 6 and 7 in rows_list (index 0 is header)
        row_6 = rows_list[6]
        row_7 = rows_list[7]
        
        self.play(
            row_6.animate.set_color("#FF0000"),
            row_7.animate.set_color("#FF0000"),
            run_time=1
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Mathematical truth requires more than just a good start.
        # Fade out the table and display the final message: 'Proof > Intuition' in bright cyan (#00FFFF).
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW),
        )
        
        final_msg = Text("Proof > Intuition", font_size=40, color="#00FFFF")
        # Resolve Issue 30: Place final_msg in A1-B6 to avoid obstruction
        self.place_in_area(final_msg, 'A1', 'B6', scale_factor=0.9)
        
        self.play(
            FadeOut(table),
            Write(final_msg),
            run_time=1.5
        )
        self.wait(4)
