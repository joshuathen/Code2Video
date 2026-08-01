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
        # Initial Setup
        title = "The Bridge: From Discrete to Continuous"
        lines = [
            "Discrete variables have separate, countable outcomes.",
            "Continuous variables like time approach infinite possibilities.",
            "Histogram bars eventually smooth into a continuous curve."
        ]
        self.setup_layout(title, lines)
        
        # Colors
        BLUE_COLOR = "#58C4DD"
        YELLOW_COLOR = "#FFFF00"
        
        # Helper function for PDF-like height
        def get_height(x):
            return 2.5 * np.exp(-(x**2) / 1.5)
            
        # Helper function to create a VGroup of bars
        def create_bars(n):
            bars = VGroup()
            total_width = 4.5
            width = total_width / n
            start_x = -total_width / 2 + width / 2
            for i in range(n):
                x = start_x + i * width
                h = get_height(x)
                # Ensure height is at least slightly visible
                h = max(h, 0.05)
                bar = Rectangle(
                    width=width, 
                    height=h, 
                    fill_opacity=0.8, 
                    fill_color=BLUE_COLOR, 
                    stroke_width=0.5 if n < 20 else 0.1, 
                    stroke_color=WHITE
                )
                # Bottom-center positioning logic
                bar.move_to([x, h/2, 0])
                bars.add(bar)
            return bars

        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.lecture[0].set_color(BLUE_COLOR)
        
        # Create initial 5 bars
        bars_obj = create_bars(5)
        # Position using area grid as requested by VideoCritic (Issues 20, 21)
        self.place_in_area(bars_obj, "B2", "F6", scale_factor=0.8)
        # Fix baseline to row F to avoid "floating"
        baseline_y = self.grid["F1"][1]
        bars_obj.align_to(np.array([0, baseline_y, 0]), DOWN)
        
        self.play(FadeIn(bars_obj))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Update lecture line colors
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(BLUE_COLOR)
        )
        
        # Store center and bottom alignment to maintain consistency during update
        ref_center_x = bars_obj.get_center()[0]
        
        # Tracker for bar count
        n_tracker = ValueTracker(5)
        
        def update_bars_func(m):
            n = int(n_tracker.get_value())
            if not hasattr(m, "last_n") or m.last_n != n:
                new_bars = create_bars(n)
                # Keep the same horizontal center and baseline
                new_bars.move_to([ref_center_x, 0, 0])
                new_bars.align_to(np.array([0, baseline_y, 0]), DOWN)
                m.become(new_bars)
                m.last_n = n

        # Smoothly increase bar density
        bars_obj.add_updater(update_bars_func)
        self.play(n_tracker.animate.set_value(50), run_time=4, rate_func=smooth)
        bars_obj.remove_updater(update_bars_func)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Update lecture line colors
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW_COLOR)
        )
        
        # Create smooth yellow curve (PDF)
        curve = FunctionGraph(
            lambda x: get_height(x),
            x_range=[-2.25, 2.25],
            color=YELLOW_COLOR,
            stroke_width=4
        )
        
        # Scale and move curve to match the histogram's final footprint
        curve.stretch_to_fit_width(bars_obj.width)
        curve.stretch_to_fit_height(bars_obj.height)
        curve.move_to(bars_obj.get_center())
        curve.align_to(bars_obj, DOWN)
        
        # Final transformation from discrete bars to smooth curve
        self.play(
            Transform(bars_obj, curve),
            run_time=2.5
        )
        self.wait(3)
