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
        # Lecture lines and Title
        title_str = "The Math: The 243 Possible Feedback Patterns"
        lines_str = [
            'Each Wordle guess returns a specific color pattern.',
            'There are three possible colors for five positions.',
            'This creates two hundred forty-three unique feedback patterns.',
            'A histogram reveals how words map to these patterns.',
            'Optimal words distribute answers evenly across all buckets.'
        ]
        
        self.setup_layout(title_str, lines_str)
        
        # Colors
        COLOR_GREY = "#787C7E"
        COLOR_YELLOW = "#C9B458"
        COLOR_GREEN = "#6AAA64"
        COLOR_WHITE = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Highlight current lecture line
        self.lecture[0].set_color(YELLOW)
        
        # Show blank 5-letter Wordle row with white outlines
        wordle_row = VGroup(*[Square(side_length=0.6, stroke_color=COLOR_WHITE) for _ in range(5)]).arrange(RIGHT, buff=0.1)
        # Issue 36 Fix: Move from B1-B6 to B2-B5
        self.place_in_area(wordle_row, "B2", "B5", scale_factor=1.0)
        
        self.play(Create(wordle_row))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transition lecture highlight
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(YELLOW))
        
        # Display 3 color squares: Grey (#787C7E), Yellow (#C9B458), and Green (#6AAA64).
        color_squares = VGroup(
            Square(side_length=0.4, fill_color=COLOR_GREY, fill_opacity=1, stroke_width=0),
            Square(side_length=0.4, fill_color=COLOR_YELLOW, fill_opacity=1, stroke_width=0),
            Square(side_length=0.4, fill_color=COLOR_GREEN, fill_opacity=1, stroke_width=0)
        ).arrange(RIGHT, buff=0.3)
        self.place_in_area(color_squares, "A2", "A5")
        
        self.play(FadeIn(color_squares))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Transition lecture highlight
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(YELLOW))
        
        # The equation '3^5 = 243' appears in bold white at the center.
        equation = Text("3^5 = 243", color=COLOR_WHITE, font_size=48, weight=BOLD)
        # Issue 35 Fix: Move from C1-C6 to C2-C5
        self.place_in_area(equation, "C2", "C5", scale_factor=1.0)
        
        self.play(Write(equation))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Transition lecture highlight
        self.play(self.lecture[2].animate.set_color(WHITE), self.lecture[3].animate.set_color(YELLOW))
        
        # A histogram with 243 thin white bars appears along the bottom.
        # Use small rectangles to represent the 243 buckets.
        bars = VGroup(*[
            Rectangle(width=0.012, height=0.1, stroke_width=0, fill_opacity=1, fill_color=COLOR_WHITE) 
            for _ in range(243)
        ]).arrange(RIGHT, buff=0.005, aligned_edge=DOWN)
        # Issue 34 Fix: Move from F1-F6 to E1-F6
        self.place_in_area(bars, "E1", "F6", scale_factor=1.0)
        
        self.play(Create(bars))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Transition lecture highlight
        self.play(self.lecture[3].animate.set_color(WHITE), self.lecture[4].animate.set_color(YELLOW))
        
        # The word 'SALET' appears, and dots distribute across the histogram bars evenly.
        word_salet = Text("SALET", font_size=36, color=COLOR_WHITE)
        self.place_in_area(word_salet, "D2", "D5")
        
        # Create dots that will distribute into buckets
        dots = VGroup(*[
            Dot(radius=0.025, color=COLOR_WHITE).move_to(word_salet.get_center()) 
            for _ in range(100)
        ])
        
        # Target heights for bars to simulate an even distribution
        target_heights = [0.8 + np.random.uniform(-0.15, 0.15) for _ in range(243)]
        
        self.play(FadeIn(word_salet))
        self.play(
            # Dots scatter towards the histogram
            LaggedStart(
                *[dot.animate.move_to(bars[np.random.randint(0, 243)].get_top() + UP * 0.1).set_opacity(0) 
                  for dot in dots], 
                lag_ratio=0.01
            ),
            # Bars grow from their current bottom edge
            *[bars[i].animate.stretch_to_fit_height(target_heights[i], about_edge=DOWN) for i in range(243)],
            run_time=3,
            rate_func=linear
        )
        self.remove(dots)
        self.wait(2)
