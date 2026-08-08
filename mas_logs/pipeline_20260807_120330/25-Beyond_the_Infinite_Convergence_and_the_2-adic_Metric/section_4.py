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

class Section4Scene(TeachingScene):
    def construct(self):
        # Title and Lecture Lines
        title = "Summation Reimagined: Why 1 + 2 + 4 + ... = -1"
        lines = [
            "If terms approach zero, the series may converge.",
            "Use the geometric series formula with ratio 2.",
            "The algebraic result is exactly negative one.",
            "Infinite binary carries to the left represent -1.",
            "In 2-adic space, this sum is perfectly stable."
        ]
        
        self.setup_layout(title, lines)
        
        # Colors for matching
        COLOR_1 = "#AAAAAA" # Grey for initial zeros
        COLOR_2 = "#FFFF00" # Yellow for carries/terms
        COLOR_3 = "#00FF00" # Green for filled sequence
        COLOR_4 = "#FFFFFF" # White for -1
        COLOR_5 = "#88FF88" # Light Green for formula
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_1))
        
        # Binary sequence of 0s. 
        # Using column 5 to use the right side more effectively (Issue 27)
        bit_pos = ["F5", "E5", "D5", "C5", "B5"]
        bits = VGroup(*[Text("0", color=COLOR_1) for _ in range(5)])
        for i, pos in enumerate(bit_pos):
            self.place_at_grid(bits[i], pos, scale_factor=0.8)
        
        self.play(FadeIn(bits, lag_ratio=0.1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(COLOR_2))
        
        # Represent adding 1, 2, 4
        # We'll flip the bottom bits one by one.
        for i in range(3):
            carry_dot = Dot(color=COLOR_2).scale(0.5)
            # Start slightly below the bit
            start_pos = self.grid[bit_pos[i]] + DOWN * 0.5
            carry_dot.move_to(start_pos)
            
            new_bit = Text("1", color=COLOR_2)
            self.place_at_grid(new_bit, bit_pos[i], scale_factor=0.8)
            
            self.play(
                carry_dot.animate.move_to(self.grid[bit_pos[i]]),
                run_time=0.4
            )
            self.play(
                ReplacementTransform(bits[i], new_bit),
                FadeOut(carry_dot),
                run_time=0.3
            )
            bits[i] = new_bit # Update reference
            
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(COLOR_3))
        
        # Fill the sequence with green 1s and ellipsis at the top.
        green_bits = VGroup()
        for i in range(5):
            gb = Text("1", color=COLOR_3)
            self.place_at_grid(gb, bit_pos[i], scale_factor=0.8)
            green_bits.add(gb)
        
        # Fixed ellipsis position to A5 (Issue 27)
        ellipsis = Text("...", color=COLOR_3)
        self.place_at_grid(ellipsis, "A5", scale_factor=0.8)
        
        self.play(
            ReplacementTransform(bits, green_bits),
            FadeIn(ellipsis),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(COLOR_4))
        
        # Transform the string into -1
        full_string = VGroup(ellipsis, green_bits)
        minus_one = Text("-1", color=COLOR_4, weight=BOLD)
        
        # Fixed position for minus_one: D5-F5 (Issue 26)
        self.place_in_area(minus_one, "D5", "F5", scale_factor=1.2)
        
        self.play(
            ReplacementTransform(full_string, minus_one),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(COLOR_5))
        
        # Formula S = 1 / (1 - 2) = -1
        # Use MathTex for mathematical notation
        formula = MathTex("S = \\frac{1}{1 - 2} = -1", color=COLOR_5)
        
        # Fixed formula position to A1-B3 (Issue 25)
        self.place_in_area(formula, "A1", "B3", scale_factor=0.8)
        
        self.play(Write(formula))
        self.wait(2)
