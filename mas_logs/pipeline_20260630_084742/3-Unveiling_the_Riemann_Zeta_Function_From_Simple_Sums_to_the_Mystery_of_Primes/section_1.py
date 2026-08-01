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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            'Imagine adding an infinite list of numbers together.', 
            'A runner covers half of a one-meter path.', 
            'Next, they cover half of the remaining gap.', 
            'The bar fills as the runner continues forever.', 
            'This infinite process reaches a finite total of one.'
        ]
        self.setup_layout("Prerequisite: The Concept of Infinite Series", lecture_lines)
        
        # Color definitions
        COLOR_BAR = "#FFFFFF"
        COLOR_RUNNER = "#00FF00"
        COLOR_FILL = "#FFFF00"
        COLOR_TEXT = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_TEXT))
        
        # Sequence of fractions representation
        fractions_seq = Text("1/2 + 1/4 + 1/8 + 1/16 + ...", font_size=24, color=COLOR_TEXT)
        self.place_in_area(fractions_seq, "A2", "A5")
        self.play(Write(fractions_seq))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(COLOR_RUNNER))
        
        # Main visual bar
        bar_outline = Rectangle(width=4.0, height=0.6, color=COLOR_BAR)
        self.place_in_area(bar_outline, "C2", "C5")
        
        # Problem: 'bar_label' centered relative to the bar (Issue 37)
        bar_label = Text("1 meter bar", font_size=24, color=COLOR_BAR)
        self.place_in_area(bar_label, "B2", "B5", scale_factor=0.8)
        
        self.play(Create(bar_outline), Write(bar_label))
        
        # Runner icon
        runner = Triangle(color=COLOR_RUNNER, fill_opacity=1).scale(0.15)
        runner.rotate(-90 * DEGREES)
        
        # Position runner at start (left of bar)
        left_x = bar_outline.get_left()[0]
        right_x = bar_outline.get_right()[0]
        bar_y = bar_outline.get_center()[1]
        width = 4.0
        
        runner.move_to([left_x, bar_y + 0.6, 0])
        self.play(FadeIn(runner))
        
        # First step: covers half (1/2)
        target_x_half = left_x + 0.5 * width
        rect_half = Rectangle(width=0.5 * width, height=0.6, fill_color=COLOR_FILL, fill_opacity=0.8, stroke_width=0)
        rect_half.move_to([left_x + 0.25 * width, bar_y, 0])
        
        self.play(
            runner.animate.move_to([target_x_half, bar_y + 0.6, 0]),
            FadeIn(rect_half),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(COLOR_RUNNER))
        
        # Second step: half of remaining (1/4)
        target_x_quarter = target_x_half + 0.25 * width
        rect_quarter = Rectangle(width=0.25 * width, height=0.6, fill_color=COLOR_FILL, fill_opacity=0.8, stroke_width=0)
        rect_quarter.move_to([target_x_half + 0.125 * width, bar_y, 0])
        
        self.play(
            runner.animate.move_to([target_x_quarter, bar_y + 0.6, 0]),
            FadeIn(rect_quarter),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(COLOR_FILL))
        
        # Iterative steps to show continuous filling
        steps = [0.125, 0.0625, 0.03125]
        current_x = target_x_quarter
        
        for step_size in steps:
            move_dist = step_size * width
            target_x = current_x + move_dist
            fill_rect = Rectangle(width=move_dist, height=0.6, fill_color=COLOR_FILL, fill_opacity=0.8, stroke_width=0)
            fill_rect.move_to([current_x + move_dist/2, bar_y, 0])
            
            self.play(
                runner.animate.move_to([target_x, bar_y + 0.6, 0]),
                FadeIn(fill_rect),
                run_time=0.5
            )
            current_x = target_x

        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(COLOR_TEXT))
        
        # Runner reaches the end and the sum is finalized
        final_gap = right_x - current_x
        final_rect = Rectangle(width=final_gap, height=0.6, fill_color=COLOR_FILL, fill_opacity=0.8, stroke_width=0)
        final_rect.move_to([current_x + final_gap/2, bar_y, 0])
        
        self.play(
            runner.animate.move_to([right_x, bar_y + 0.6, 0]),
            FadeIn(final_rect),
            run_time=1
        )
        
        # Result sum display (Issue 38)
        result_formula = Text("Sum (1/2^n) = 1", color=COLOR_TEXT, font_size=32)
        self.place_in_area(result_formula, "E2", "E5", scale_factor=0.9)
        
        self.play(Write(result_formula))
        self.wait(2)
