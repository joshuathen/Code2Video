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
        # Title and Lecture Lines
        title = "The Revelation: Emergence of the Normal Distribution"
        lines = [
            "As we collect more samples, something magical happens.",
            "The graph transforms into a smooth, symmetrical bell shape.",
            "This beautiful curve is called the Normal Distribution.",
            "It emerges regardless of the original population's messy shape.",
            "This phenomena is the famous Central Limit Theorem."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # As we collect more samples, something magical happens.
        self.lecture[0].set_color(YELLOW)
        
        # Dots cluster representing initial messy data
        dots = VGroup(*[Dot(radius=0.04, color=WHITE, fill_opacity=0.6) for _ in range(100)])
        for dot in dots:
            dot.move_to([np.random.uniform(-1.5, 1.5), np.random.uniform(-1, 1), 0])
        
        # Anchor visualization to grid
        self.place_in_area(dots, "B1", "E6", scale_factor=0.8)
        
        self.play(FadeIn(dots, shift=UP*0.3), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The graph transforms into a smooth, symmetrical bell shape.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Rough Cyan histogram
        bar_heights = [0.4, 1.0, 1.8, 2.5, 2.8, 2.5, 1.8, 1.0, 0.4]
        bars = VGroup()
        bar_width = 0.5
        for i, h in enumerate(bar_heights):
            rect = Rectangle(
                width=bar_width - 0.05, 
                height=h, 
                fill_color="#00FFFF", 
                fill_opacity=0.6, 
                stroke_width=1
            )
            rect.move_to([(i - len(bar_heights)/2) * bar_width, 0, 0], aligned_edge=DOWN)
            bars.add(rect)
        
        # Fix Issue 28: Anchor histogram to grid
        self.place_in_area(bars, "B1", "E6", scale_factor=0.8)
        
        self.play(
            ReplacementTransform(dots, bars),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # This beautiful curve is called the Normal Distribution.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Smoother and taller bars
        smooth_heights = [0.2, 0.4, 0.8, 1.3, 1.9, 2.6, 3.2, 3.5, 3.2, 2.6, 1.9, 1.3, 0.8, 0.4, 0.2]
        smooth_bars = VGroup()
        s_bar_width = 0.3
        for i, h in enumerate(smooth_heights):
            rect = Rectangle(
                width=s_bar_width - 0.02, 
                height=h, 
                fill_color="#00FFFF", 
                fill_opacity=0.7, 
                stroke_width=0.5
            )
            rect.move_to([(i - len(smooth_heights)/2) * s_bar_width, 0, 0], aligned_edge=DOWN)
            smooth_bars.add(rect)
            
        # Keep consistent anchoring
        self.place_in_area(smooth_bars, "B1", "E6", scale_factor=0.8)
        
        self.play(
            ReplacementTransform(bars, smooth_bars),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # It emerges regardless of the original population's messy shape.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Gold bell curve overlay
        normal_curve = FunctionGraph(
            lambda x: 3.6 * np.exp(-(x**2) / (2 * 1.3**2)),
            x_range=[-3.5, 3.5],
            color="#FFD700"
        )
        
        # Fix Issue 29: Anchor normal curve to grid
        self.place_in_area(normal_curve, "B1", "E6", scale_factor=0.8)
        # Ensure it sits on the same baseline as the histogram
        normal_curve.align_to(smooth_bars, DOWN)
        
        self.play(Create(normal_curve), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # This phenomena is the famous Central Limit Theorem.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Pulse bell curve brightness via stroke width highlight
        self.play(
            normal_curve.animate.set_stroke(width=10),
            run_time=0.6,
            rate_func=there_and_back
        )
        self.play(
            normal_curve.animate.set_stroke(width=10),
            run_time=0.6,
            rate_func=there_and_back
        )
        self.wait(3)
