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

class Section3Scene(TeachingScene):
    def construct(self):
        # Layout data
        title_text = "Step 1: The Oracle (The Negative Flip)"
        lecture_lines = [
            'The Oracle identifies the correct answer among possibilities.',
            'It flips the phase of the target state specifically.',
            'Mathematically, it multiplies target amplitude by negative one.',
            'The target bar flips below the horizontal axis.',
            'All other amplitudes remain unchanged and upright.'
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Color definitions
        ORANGE = "#FF8C00"
        CYAN = "#00FFFF"
        BLUE_B = "#58C4DD"
        
        # Define chart parameters
        bar_count = 8
        target_idx = 5  # Symbolizes Box #42
        bar_height = 1.4
        bar_width = 0.4
        bar_spacing = 0.2
        
        # Create bars
        bars = VGroup()
        for i in range(bar_count):
            bar = Rectangle(
                width=bar_width, 
                height=bar_height, 
                fill_color=BLUE_B, 
                fill_opacity=0.8,
                stroke_width=1
            )
            bars.add(bar)
        
        bars.arrange(RIGHT, buff=bar_spacing)
        
        # Position bars within the designated grid area (Issue 39 fix)
        self.place_in_area(bars, 'B1', 'F6', scale_factor=1.0)
        
        # Establish the baseline (horizontal axis)
        baseline_y = bars.get_bottom()[1]
        baseline = Line(
            [bars.get_left()[0] - 0.3, baseline_y, 0],
            [bars.get_right()[0] + 0.3, baseline_y, 0],
            color=WHITE,
            stroke_width=2
        )
        
        # Generate labels for state indices
        labels = VGroup()
        for i, bar in enumerate(bars):
            text = "42" if i == target_idx else str(i)
            # Position labels slightly further down to avoid overlap during flip
            lbl = Text(text, font_size=16).next_to(bar, DOWN, buff=0.15)
            labels.add(lbl)
            
        chart = VGroup(bars, baseline, labels)

        # === Animation for Lecture Line 1 ===
        # The Oracle identifies the correct answer among possibilities.
        self.play(self.lecture[0].animate.set_color(ORANGE))
        self.add(chart)
        self.play(bars[target_idx].animate.set_color(ORANGE))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # It flips the phase of the target state specifically.
        self.play(self.lecture[1].animate.set_color(ORANGE))
        # Asset integration (Issue 31) and Position fix (Issue 38)
        oracle_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/oracle.svg").set_color(ORANGE).scale(0.3)
        oracle_text = Text("Oracle (U_w)", font_size=24, color=ORANGE)
        oracle_group = VGroup(oracle_icon, oracle_text).arrange(RIGHT, buff=0.3)
        self.place_at_grid(oracle_group, 'A4', scale_factor=0.8)
        
        self.play(FadeIn(oracle_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Mathematically, it multiplies target amplitude by negative one.
        self.play(self.lecture[2].animate.set_color(ORANGE))
        target_bar = bars[target_idx]
        # Invert the target bar vertically across the baseline
        self.play(
            target_bar.animate.shift(DOWN * bar_height)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # The target bar flips below the horizontal axis.
        self.play(self.lecture[3].animate.set_color(CYAN))
        
        # Show and shift average line
        # Initial average: 1.4. New average after one flip: (7*1.4 - 1.4)/8 = 1.05
        avg_line_y = baseline_y + bar_height
        avg_line = DashedLine(
            [baseline.get_left()[0], avg_line_y, 0],
            [baseline.get_right()[0], avg_line_y, 0],
            color=CYAN,
            stroke_width=3
        )
        
        avg_label = Text("Average", font_size=14, color=CYAN)
        avg_label.next_to(avg_line, RIGHT, buff=0.1)
        
        self.play(Create(avg_line), Write(avg_label))
        # Relative shift to new average position
        shift_amount = bar_height * (1.0 - 1.05/1.4)
        self.play(
            avg_line.animate.shift(DOWN * shift_amount),
            avg_label.animate.shift(DOWN * shift_amount)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # All other amplitudes remain unchanged and upright.
        self.play(self.lecture[4].animate.set_color(ORANGE))
        # Pulse the target bar to emphasize the modification
        self.play(
            target_bar.animate.scale(1.2),
            rate_func=there_and_back,
            run_time=0.6
        )
        self.play(
            target_bar.animate.scale(1.2),
            rate_func=there_and_back,
            run_time=0.6
        )
        self.wait(2)
