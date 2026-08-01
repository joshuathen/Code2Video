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
        # Initialize the layout with title and lecture lines
        self.setup_layout(
            "Conclusion: The Quadratic Speedup", 
            [
                "Grover's Algorithm provides a powerful quadratic speedup.", 
                "One million items searched in one thousand steps.", 
                "Quantum search is more efficient than classical methods."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Display a comparison: 'Classical: 500,000 steps' vs 'Grover: 1,000 steps' in white (#FFFFFF)
        classic_text = Text("Classical: 500,000 steps", font_size=24, color=WHITE)
        grover_text = Text("Grover: 1,000 steps", font_size=24, color=WHITE)
        comparison = VGroup(classic_text, grover_text).arrange(DOWN, buff=0.5)
        
        self.place_in_area(comparison, "A1", "B6")
        
        self.play(
            FadeIn(comparison),
            self.lecture[0].animate.set_color(YELLOW)
        )
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Show the target bar probability reaching nearly 100% (height 1.0).
        # We'll use a frame to represent the 0-1 probability scale.
        bar_height_full = 2.5
        bar_frame = Rectangle(height=bar_height_full, width=1.5, color=WHITE)
        
        # The filling bar starting very small
        bar_fill = Rectangle(height=0.01, width=1.5, fill_opacity=0.8, color=BLUE, stroke_width=0)
        bar_fill.align_to(bar_frame, DOWN)
        
        one_label = Text("1.0", font_size=20, color=WHITE).next_to(bar_frame, UP, buff=0.1)
        zero_label = Text("0.0", font_size=20, color=WHITE).next_to(bar_frame, DOWN, buff=0.1)
        prob_title = Text("Success Probability", font_size=18, color=WHITE).next_to(bar_frame, LEFT, buff=0.3)
        
        bar_group = VGroup(bar_frame, bar_fill, one_label, zero_label, prob_title)
        self.place_in_area(bar_group, "C1", "E6")
        
        # Highlight line 2 and update colors
        self.play(
            FadeIn(bar_frame), 
            FadeIn(one_label), 
            FadeIn(zero_label), 
            FadeIn(prob_title),
            FadeIn(bar_fill),
            self.lecture[1].animate.set_color(YELLOW),
            self.lecture[0].animate.set_color(WHITE)
        )
        
        # ValueTracker for the bar height to show gradual increase
        h_tracker = ValueTracker(0.01)
        bar_fill.add_updater(
            lambda m: m.stretch_to_fit_height(h_tracker.get_value()).align_to(bar_frame, DOWN)
        )
        
        # Growth animation representing finding the pixel with nearly 100% success
        self.play(
            h_tracker.animate.set_value(bar_height_full * 0.98), 
            run_time=2.0, 
            rate_func=smooth
        )
        self.wait(1)
        bar_fill.remove_updater(bar_fill.updaters[0])

        # === Animation for Lecture Line 3 ===
        # Fade in text 'Pixel found with high probability!' in green (#00FF00).
        success_text = Text("Pixel found with high probability!", font_size=26, color="#00FF00")
        self.place_in_area(success_text, "F1", "F6")
        
        self.play(
            FadeIn(success_text),
            self.lecture[2].animate.set_color(YELLOW),
            self.lecture[1].animate.set_color(WHITE)
        )
        self.wait(3)