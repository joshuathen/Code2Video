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
    def construct(self):
        # Setup title and lecture lines
        title_text = "The 'Aha!' Moment: Identifying the Culprit"
        lecture_lines = [
            "A bit flip causes specific parity checks to fail.",
            "Sum the indices of all failed parity checks.",
            "This total points directly to the erroneous bit.",
            "If the sum is three, position three is wrong.",
            "Flip the bit back to restore the original data."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Defining Colors
        FAIL_COLOR = "#FF0000"
        SUCCESS_COLOR = "#00FF00"
        NEUTRAL_COLOR = "#FFFFFF"

        # Create Parity Circles
        # We model Hamming(7,4) using three overlapping circles P1, P2, P4
        p1_circle = Circle(radius=1.2, color=FAIL_COLOR, stroke_width=4)
        p2_circle = Circle(radius=1.2, color=FAIL_COLOR, stroke_width=4)
        p4_circle = Circle(radius=1.2, color=SUCCESS_COLOR, stroke_width=4)

        # Position circles (centers)
        self.place_at_grid(p1_circle, "C2")
        self.place_at_grid(p2_circle, "C4")
        self.place_at_grid(p4_circle, "E3")

        # Labels for the circles
        p1_label = Text("P1", font_size=20, color=NEUTRAL_COLOR)
        p2_label = Text("P2", font_size=20, color=NEUTRAL_COLOR)
        p4_label = Text("P4", font_size=20, color=NEUTRAL_COLOR)
        
        self.place_at_grid(p1_label, "B1")
        self.place_at_grid(p2_label, "B5")
        # Fix for Issue 41
        self.place_at_grid(p4_label, "E4", scale_factor=0.8)

        # Bit Indices inside the circles
        bits = VGroup()
        pos_map = {
            "1": "B2", # P1 only
            "2": "B4", # P2 only
            "3": "C3", # P1 & P2
            "4": "F3", # P4 only
            "5": "D2", # P1 & P4
            "6": "D4", # P2 & P4
            "7": "D3"  # All
        }
        
        bit_mobjects = {}
        for idx, pos in pos_map.items():
            b = Text(idx, font_size=24, color=NEUTRAL_COLOR)
            self.place_at_grid(b, pos)
            bit_mobjects[idx] = b
            bits.add(b)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(FAIL_COLOR))
        self.play(
            Create(p1_circle),
            Create(p2_circle),
            Create(p4_circle),
            Write(bits),
            Write(p1_label),
            Write(p2_label),
            Write(p4_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Use Text instead of MathTex to avoid FileNotFoundError: 'latex'
        math_text = Text("Error = P1 + P2 = 1 + 2 = 3", font_size=24, color=NEUTRAL_COLOR)
        # Fix for Issue 42
        self.place_in_area(math_text, "A1", "A6", scale_factor=0.7)
        
        self.play(self.lecture[1].animate.set_color(NEUTRAL_COLOR))
        self.play(Write(math_text))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(NEUTRAL_COLOR))
        highlight_box = SurroundingRectangle(bit_mobjects["3"], color=FAIL_COLOR, buff=0.1)
        self.play(Create(highlight_box))
        self.play(bit_mobjects["3"].animate.set_color(FAIL_COLOR))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(NEUTRAL_COLOR))
        self.play(Flash(bit_mobjects["3"], color=FAIL_COLOR, line_length=0.2))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(SUCCESS_COLOR))
        
        success_text = Text("Data recovered successfully!", font_size=24, color=NEUTRAL_COLOR)
        # Fix for Issue 40
        self.place_in_area(success_text, "F1", "F6", scale_factor=0.5)
        
        self.play(
            p1_circle.animate.set_color(SUCCESS_COLOR),
            p2_circle.animate.set_color(SUCCESS_COLOR),
            bit_mobjects["3"].animate.set_color(SUCCESS_COLOR),
            Uncreate(highlight_box),
            Write(success_text)
        )
        self.wait(2)
