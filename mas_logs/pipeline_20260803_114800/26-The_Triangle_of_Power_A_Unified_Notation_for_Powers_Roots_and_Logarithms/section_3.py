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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Introducing the Triangle of Power", [
            "Meet the Triangle of Power: a unified mathematical map.",
            "The base sits at the bottom-left vertex.",
            "The exponent climbs to the top peak.",
            "The final result rests at the bottom-right vertex.",
            "One single shape replaces three different notations."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Meet the Triangle of Power: a unified mathematical map.
        self.lecture[0].set_color(WHITE)
        
        # Define vertices according to Critic fixes in Issues 26, 27, 28
        # v_bl at E3 (Issue 26)
        # v_br at E6 (Issue 28)
        # v_top at center of B4-B5 (Issue 27)
        v_bl = self.grid['E3']
        v_br = self.grid['E6']
        v_top = (self.grid['B4'] + self.grid['B5']) / 2
        
        triangle = Polygon(v_bl, v_br, v_top, color=WHITE, stroke_width=4)
        
        self.play(Create(triangle), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The base sits at the bottom-left vertex.
        self.lecture[1].set_color("#0000FF")
        base_val = MathTex("3", color="#0000FF")
        # Fix from Issue 26: Use E3, scale 1.2
        self.place_at_grid(base_val, 'E3', scale_factor=1.2)
        
        self.play(Write(base_val))
        self.play(Indicate(base_val, color="#0000FF"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The exponent climbs to the top peak.
        self.lecture[2].set_color("#00FF00")
        exp_val = MathTex("2", color="#00FF00")
        # Fix from Issue 27: Use area B4-B5, scale 1.2
        self.place_in_area(exp_val, 'B4', 'B5', scale_factor=1.2)
        
        self.play(Write(exp_val))
        self.play(Indicate(exp_val, color="#00FF00"))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # The final result rests at the bottom-right vertex.
        self.lecture[3].set_color("#FF0000")
        res_val = MathTex("9", color="#FF0000")
        # Fix from Issue 28: Use E6, scale 1.2
        self.place_at_grid(res_val, 'E6', scale_factor=1.2)
        
        self.play(Write(res_val))
        self.play(Indicate(res_val, color="#FF0000"))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # One single shape replaces three different notations.
        self.lecture[4].set_color(WHITE)
        
        # Entire triangle pulses with soft white light
        self.play(Indicate(triangle, color=WHITE, scale_factor=1.1))
        self.wait(2)
