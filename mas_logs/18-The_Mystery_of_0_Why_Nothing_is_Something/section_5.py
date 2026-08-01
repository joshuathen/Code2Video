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

class Section5Scene(TeachingScene):
    def construct(self):
        # Setup Title and Lecture Lines
        lecture_lines = [
            "The combinations formula helps us choose items from sets.",
            "Let's try choosing all items from a group.",
            "If zero factorial was zero, we'd divide by zero.",
            "Setting it to one keeps our formulas working perfectly.",
            "This definition ensures all of mathematics remains stable."
        ]
        self.setup_layout("The Big Picture: Consistency is King", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Combinations formula: n choose k = n! / (k!(n-k)!)
        formula1 = Text("n choose k = n! / (k! (n-k)!)", color="#FFFFFF", font_size=24)
        self.place_at_grid(formula1, "B3", scale_factor=1.0)
        
        self.play(
            Write(formula1),
            self.lecture[0].animate.set_color("#FFFFFF"),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Substitute k=n: n choose n = n! / (n! 0!)
        formula2 = Text("n choose n = n! / (n! 0!)", color="#FFFFFF", font_size=24)
        self.place_at_grid(formula2, "B3", scale_factor=1.0)
        
        self.play(
            Transform(formula1, formula2),
            self.lecture[1].animate.set_color("#FFFFFF"),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Flash the '0!' in the denominator with a red warning color
        # Since we use Text, we create a highlight over the "0!" part
        warning_box = SurroundingRectangle(formula1, color="#FF0000", buff=0.1)
        zero_fact_label = Text("0!", color="#FF0000", font_size=28)
        self.place_at_grid(zero_fact_label, "B5", scale_factor=1.0)
        
        self.play(
            Create(warning_box),
            Write(zero_fact_label),
            self.lecture[2].animate.set_color("#FF0000"),
            run_time=1
        )
        self.play(Flash(zero_fact_label, color="#FF0000"), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Replace '0!' with '1' and simplify to '1'
        formula3 = Text("n choose n = n! / (n! 1) = 1", color="#00FF00", font_size=24)
        self.place_at_grid(formula3, "B3", scale_factor=1.0)
        
        self.play(
            FadeOut(warning_box),
            FadeOut(zero_fact_label),
            Transform(formula1, formula3),
            self.lecture[3].animate.set_color("#00FF00"),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Visual of a stone bridge [Asset: bridge.svg replaced with RoundedRectangle for compatibility]
        # Issue 22: Integrate bridge.svg
        bridge = RoundedRectangle(fill_color="#808080", stroke_color=WHITE, fill_opacity=1.0)
        self.place_in_area(bridge, "D1", "F6", scale_factor=1.8)
        
        keystone_label = Text("0! = 1", color="#FFFF00", font_size=28)
        # Position keystone in the center of the bridge area
        self.place_at_grid(keystone_label, "E4", scale_factor=0.8)
        
        self.play(
            DrawBorderThenFill(bridge),
            Write(keystone_label),
            self.lecture[4].animate.set_color("#FFFF00"),
            run_time=2
        )
        self.play(Indicate(keystone_label, color="#FFFF00"), run_time=1)
        self.wait(3)
