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
        # Fetching data from shared state
        title = "The Quantum Speedup Summary"
        lines = [
            "Grover's algorithm provides a significant quadratic speedup over classical.",
            "A million items take only one thousand quantum iterations.",
            "Quantum mechanics finds the needle in the haystack much faster."
        ]
        
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # Display a split-screen comparison: 'Classical: O(N)' on left, 'Grover: O(√N)' on right.
        line1_color = BLUE_A
        self.lecture[0].set_color(line1_color)
        
        # Assets: Haystack icon
        haystack = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/haystack.svg")
        self.place_at_grid(haystack, "A4", scale_factor=0.6)
        
        # Use Text instead of MathTex to avoid LaTeX dependency
        classical_label = VGroup(
            Text("Classical:", font_size=24, color=WHITE),
            Text("O(N)", font_size=36, color=line1_color)
        ).arrange(DOWN, buff=0.2)
        
        grover_label = VGroup(
            Text("Grover:", font_size=24, color=WHITE),
            # Use unicode square root symbol
            Text("O(√N)", font_size=36, color=line1_color)
        ).arrange(DOWN, buff=0.2)
        
        # L008: Using place_in_area to balance visuals
        # Addressing Issue 38: Move to Row A-B
        self.place_in_area(classical_label, "A2", "B3", scale_factor=0.8)
        self.place_in_area(grover_label, "A5", "B6", scale_factor=0.8)
        
        # Visual separator
        separator = Line(self.grid["A4"] + DOWN*0.2, self.grid["F4"] + DOWN*0.5, color=GREY_E)
        
        self.play(
            FadeIn(haystack),
            FadeIn(classical_label),
            FadeIn(grover_label),
            Create(separator),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show a counter using DecimalNumber (standard Manim, num_decimal_places=0 for integer effect)
        line2_color = "#ADD8E6"
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(line2_color)
        
        # Value trackers for the counter effect
        classical_val = ValueTracker(1)
        grover_val = ValueTracker(1)
        
        # Use DecimalNumber with mob_class=Text to avoid LaTeX dependency
        classical_counter = DecimalNumber(1, num_decimal_places=0, group_with_commas=True, font_size=32, color=line2_color, mob_class=Text)
        classical_counter.add_updater(lambda d: d.set_value(classical_val.get_value()))
        
        grover_counter = DecimalNumber(1, num_decimal_places=0, group_with_commas=True, font_size=32, color=line2_color, mob_class=Text)
        grover_counter.add_updater(lambda d: d.set_value(grover_val.get_value()))
        
        classical_unit = Text("steps", font_size=20, color=line2_color)
        grover_unit = Text("steps", font_size=20, color=line2_color)
        
        classical_group = VGroup(classical_counter, classical_unit).arrange(DOWN, buff=0.1)
        grover_group = VGroup(grover_counter, grover_unit).arrange(DOWN, buff=0.1)
        
        # Addressing Issue 39: Positioning C2-D3 and C5-D6 with scale 0.8
        self.place_in_area(classical_group, "C2", "D3", scale_factor=0.8)
        self.place_in_area(grover_group, "C5", "D6", scale_factor=0.8)
        
        self.add(classical_group, grover_group)
        
        # Animate counting
        self.play(
            classical_val.animate.set_value(1000000),
            grover_val.animate.set_value(1000),
            run_time=3,
            rate_func=linear
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Display the text 'Quadratic Speedup Achieved!' in gold (#FFD700) alongside needle icon
        line3_color = "#FFD700"
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(line3_color)
        
        # Assets: Needle icon
        needle = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/needle.svg")
        self.place_at_grid(needle, "E1", scale_factor=0.6)
        
        speedup_text = Text("Quadratic Speedup Achieved!", font_size=32, color=line3_color)
        # Addressing Issue 40: Positioning E2-F6 with scale 0.7
        self.place_in_area(speedup_text, "E2", "F6", scale_factor=0.7)
        
        # Flash effect
        flash = Flash(speedup_text, color=line3_color, line_length=0.3, num_lines=12, flash_radius=2)
        
        self.play(
            FadeIn(needle),
            Write(speedup_text),
            run_time=1.5
        )
        self.play(flash)
        self.wait(3)
