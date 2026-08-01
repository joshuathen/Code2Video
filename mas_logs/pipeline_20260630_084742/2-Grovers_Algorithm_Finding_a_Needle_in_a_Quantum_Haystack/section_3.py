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
        # Define lecture lines with bullets
        lecture_content = [
            "- The Oracle identifies the target state using phase inversion.",
            "- It flips the target's amplitude while others stay positive.",
            "- This step marks the answer without revealing its index.",
            "- Geometrically, it reflects the state across the non-target subspace.",
            "- Only the target's sign is changed by the Oracle."
        ]
        
        self.setup_layout("Step 1: The Oracle (Phase Inversion)", lecture_content)
        
        # Define Colors
        CYAN = "#00FFFF"
        GOLD = "#FFD700"
        MAGENTA = "#FF00FF"
        RED = "#FF0000"
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(CYAN)
        
        # Visual anchor for the bars: Axis on Row C (y=0.2) to provide room for inversion
        axis_start_pos = self.grid['C1'] + LEFT * 0.5
        axis_end_pos = self.grid['C6'] + RIGHT * 0.5
        axis = Line(axis_start_pos, axis_end_pos, color=WHITE)
        self.add(axis)
        
        bars = VGroup()
        for i in range(8):
            # Each bar is 0.4 wide, 1.5 tall
            bar = Rectangle(height=1.5, width=0.35, fill_opacity=1, color=CYAN, stroke_width=1)
            # Position manually along the axis line using grid logic
            bar_x = axis.get_start()[0] + (i + 0.5) * (axis.get_length() / 8)
            bar.move_to([bar_x, axis.get_start()[1] + 0.75, 0])
            bars.add(bar)
            
        self.play(Create(bars), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(GOLD)
        
        # Highlight target bar (6th bar, index 5)
        target_bar = bars[5]
        
        # Oracle Box positioned across B1-E6 to frame the entire operation (Fix for Issue 26/27)
        oracle_box = RoundedRectangle(corner_radius=0.1, height=4.0, width=5.8, color=MAGENTA, fill_opacity=0.1)
        oracle_label = Text("Oracle", font_size=24, color=MAGENTA)
        oracle_group = VGroup(oracle_box, oracle_label)
        oracle_label.move_to(oracle_box.get_top() + DOWN * 0.4)
        
        self.place_in_area(oracle_group, 'B1', 'E6', scale_factor=0.9) 
        
        self.play(
            target_bar.animate.set_color(GOLD),
            FadeIn(oracle_group)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(MAGENTA)
        
        # Phase Inversion: Rotate 180 degrees around the axis (about the bottom of the bar)
        # pivot is at the axis level (Row C)
        pivot_y = self.grid['C1'][1]
        pivot = [target_bar.get_x(), pivot_y, 0]
        self.play(
            Rotate(target_bar, angle=PI, axis=RIGHT, about_point=pivot),
            run_time=1.2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(RED)
        
        # The 6th bar pulses in red (#FF0000) to show the negative amplitude
        self.play(
            target_bar.animate.set_color(RED),
            rate_func=there_and_back,
            run_time=0.4
        )
        self.play(
            target_bar.animate.set_color(RED),
            rate_func=there_and_back,
            run_time=0.4
        )
        target_bar.set_color(GOLD)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(GOLD)
        
        # Fade out the "Oracle" box while keeping the 6th bar pointing downwards.
        self.play(FadeOut(oracle_group))
        self.wait(2)
