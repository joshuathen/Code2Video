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
        # Initial layout setup
        self.setup_layout(
            "The Indeterminate Crisis: 0/0 and \u221e/\u221e", 
            [
                "Direct substitution often leads to zero divided by zero.", 
                "This creates a tug-of-war between numerator and denominator.", 
                "We need a way to break this mathematical tie."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Match lecture line color to the primary animation theme (Red for alert)
        self.play(self.lecture[0].animate.set_color("#FF0000"))
        
        # Display the fraction sin(x)/x - Using Text to avoid LaTeX dependency (Errno 2 'latex')
        fraction = Text("sin(x) / x", color="#FFFFFF")
        self.place_in_area(fraction, "B3", "D4", scale_factor=1.5)
        self.play(Write(fraction))
        self.wait(1)
        
        # Transform into a red "0 / 0"
        indet_form = Text("0 / 0", color="#FF0000")
        self.place_in_area(indet_form, "B3", "D4", scale_factor=1.5)
        self.play(Transform(fraction, indet_form))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Match lecture line color to the tug-of-war ropes
        self.play(self.lecture[1].animate.set_color("#0000FF"))
        
        # Blue rope pulling the numerator towards zero
        # Start at A2, pull towards top of fraction (B3)
        rope_num = Line(self.grid["A2"], self.grid["B3"], color="#0000FF", stroke_width=8)
        label_num = Text("Numerator \u2192 0", font_size=20, color="#0000FF")
        self.place_at_grid(label_num, "A2", scale_factor=1.0)
        
        # Orange rope pulling the denominator towards infinity
        # Start at E5, pull towards bottom of fraction (D4)
        rope_den = Line(self.grid["E5"], self.grid["D4"], color="#FFA500", stroke_width=8)
        # Using unicode for arrow and infinity to avoid LaTeX dependency
        label_den = Text("Denominator \u2192 \u221e", font_size=20, color="#FFA500")
        self.place_at_grid(label_den, "E5", scale_factor=1.0)
        
        self.play(
            Create(rope_num),
            Create(rope_den),
            FadeIn(label_num),
            FadeIn(label_den)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Reset lecture color to white for the final query
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        
        # Pulsing question mark in the center of the tug-of-war
        q_mark = Text("?", font_size=120, color="#FFFFFF")
        # Position right on top of the 0/0 fraction
        self.place_in_area(q_mark, "B3", "D4")
        
        self.play(FadeIn(q_mark))
        # Pulse animation
        self.play(
            q_mark.animate.scale(1.3),
            rate_func=there_and_back,
            run_time=1.0
        )
        self.play(
            q_mark.animate.scale(1.3),
            rate_func=there_and_back,
            run_time=1.0
        )
        
        self.wait(2)
