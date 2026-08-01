from manim import *

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
        # Initialization
        title_text = "The Bridge to Pi: The Leibniz Connection"
        lecture_lines = [
            'These buckets bridge the gap between numbers and geometry.', 
            'We alternate fractions from each bucket in a series.', 
            'This algebraic pattern begins to construct a circle.', 
            'Positive and negative terms balance to find the area.', 
            'The infinite sum converges exactly to pi over four.'
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Constants
        GOLD_COLOR = "#FFD700"
        SILVER_COLOR = "#C0C0C0"
        HIGHLIGHT_COLOR = "#00BFFF" # DeepSkyBlue
        
        # === Animation for Lecture Line 1 ===
        # Leibniz formula building
        formula_parts = VGroup(
            Text("1"),          # 0
            Text("-"),          # 1
            Text("1/3"),        # 2
            Text("+"),          # 3
            Text("1/5"),        # 4
            Text("-"),          # 5
            Text("1/7"),        # 6
            Text("+"),          # 7
            Text("..."),        # 8
            Text("="),          # 9
            Text("π/4")         # 10
        ).arrange(RIGHT, buff=0.15)
        
        # Fix for Issue 35: formula_parts position
        self.place_in_area(formula_parts, 'A2', 'B5', scale_factor=0.7)
        
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        self.play(FadeIn(formula_parts, shift=UP))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Color specific terms based on their source buckets
        self.lecture[1].set_color(HIGHLIGHT_COLOR)
        
        self.play(
            formula_parts[0].animate.set_color(GOLD_COLOR),  # Denominator 1 (4k+1)
            formula_parts[4].animate.set_color(GOLD_COLOR),  # Denominator 5 (4k+1)
            formula_parts[2].animate.set_color(SILVER_COLOR), # Denominator 3 (4k+3)
            formula_parts[6].animate.set_color(SILVER_COLOR), # Denominator 7 (4k+3)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Geometry: Draw circle outline
        self.lecture[2].set_color(HIGHLIGHT_COLOR)
        
        # Define a circle outline
        circle_outline = Circle(radius=0.75, color=WHITE)
        # Fix for Issue 36: circle_outline position
        self.place_in_area(circle_outline, 'D2', 'F4', scale_factor=1.0)
        
        self.play(Create(circle_outline))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Bars and jagged area approximation
        self.lecture[3].set_color(HIGHLIGHT_COLOR)
        
        # Create visual bars to represent adding/subtracting terms
        bar_gold = Rectangle(height=1.2, width=0.4, color=GOLD_COLOR, fill_opacity=0.6)
        bar_silver = Rectangle(height=0.4, width=0.4, color=SILVER_COLOR, fill_opacity=0.6)
        bars = VGroup(bar_gold, bar_silver).arrange(RIGHT, aligned_edge=DOWN, buff=0.2)
        # Fix for Issue 37: bars position
        self.place_at_grid(bars, 'C5', scale_factor=0.5)
        
        # Jagged visualization (Star as a proxy for an irregular area)
        jagged_area = Star(n=8, inner_radius=0.4, outer_radius=0.75, color=GOLD_COLOR, fill_opacity=0.4)
        # Using same area for alignment
        self.place_in_area(jagged_area, 'D2', 'F4', scale_factor=1.0)
        
        self.play(
            GrowFromCenter(jagged_area),
            FadeIn(bars, shift=RIGHT)
        )
        
        # Visual "pulse" of adding/subtracting
        self.play(
            jagged_area.animate.scale(0.85).set_color(SILVER_COLOR),
            rate_func=there_and_back,
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The jagged shape transforms into a smooth circle
        self.lecture[4].set_color(HIGHLIGHT_COLOR)
        
        perfect_circle_fill = Circle(radius=0.75, color=HIGHLIGHT_COLOR, fill_opacity=0.5)
        self.place_in_area(perfect_circle_fill, 'D2', 'F4', scale_factor=1.0)
        
        self.play(
            ReplacementTransform(jagged_area, perfect_circle_fill),
            circle_outline.animate.set_color(HIGHLIGHT_COLOR),
            formula_parts[10].animate.set_color(HIGHLIGHT_COLOR).scale(1.2),
            FadeOut(bars)
        )
        self.wait(2)
