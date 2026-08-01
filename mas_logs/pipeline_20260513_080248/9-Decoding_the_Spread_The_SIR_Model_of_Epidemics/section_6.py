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
        # Setup layout
        title_text = "The Tipping Point: Basic Reproduction Number (R0)"
        lecture_lines = [
            "Dividing Beta by Gamma gives the number R-naught.",
            "This value predicts if a disease becomes an epidemic.",
            "If R-naught exceeds one, cases grow exponentially.",
            "Below one, the outbreak will eventually die out.",
            "It is the tipping point for public health."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        RED = "#e74c3c"
        BLUE = "#3498db"
        GREEN = "#2ecc71"
        WHITE_COLOR = "#ffffff"
        YELLOW_COLOR = "#f1c40f"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE_COLOR)
        # R0 = β / γ
        formula_text = "R₀ = \u03b2 / \u03b3" # R0 = beta / gamma
        formula = Text(formula_text, font_size=48, color=WHITE_COLOR)
        # Fix Issue 50: Use B2-B5 area for better spacing
        self.place_in_area(formula, 'B2', 'B5', scale_factor=0.9)
        
        self.play(FadeIn(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(WHITE_COLOR)
        # Move formula to top row to clear space for branching diagram
        target_pos = self.grid['A3']
        self.play(formula.animate.scale(0.8).move_to(target_pos))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(RED)
        
        # Branching Diagram for R0 = 3
        # Parent Dot
        parent_dot = Circle(radius=0.25, color=RED, fill_opacity=1)
        self.place_at_grid(parent_dot, 'C3')
        
        # Target Dots
        targets = VGroup(
            Circle(radius=0.25, color=BLUE, fill_opacity=1),
            Circle(radius=0.25, color=BLUE, fill_opacity=1),
            Circle(radius=0.25, color=BLUE, fill_opacity=1)
        )
        self.place_at_grid(targets[0], 'E2')
        self.place_at_grid(targets[1], 'E3')
        self.place_at_grid(targets[2], 'E4')
        
        # Arrows
        arrows = VGroup(
            Arrow(parent_dot.get_bottom(), targets[0].get_top(), buff=0.1, color=RED),
            Arrow(parent_dot.get_bottom(), targets[1].get_top(), buff=0.1, color=RED),
            Arrow(parent_dot.get_bottom(), targets[2].get_top(), buff=0.1, color=RED)
        )
        
        spread_label = Text("R0 = 3: Epidemic Spreads", font_size=24, color=RED)
        # Fix Issue 51: Wider area for label
        self.place_in_area(spread_label, 'F1', 'F6', scale_factor=0.8)

        self.play(FadeIn(parent_dot))
        self.play(FadeIn(targets))
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.3))
        self.play(targets.animate.set_color(RED))
        self.play(FadeIn(spread_label))
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(GREEN)
        
        # Clear previous diagram
        self.play(FadeOut(parent_dot), FadeOut(targets), FadeOut(arrows), FadeOut(spread_label))
        
        # New diagram: Recovery before spread
        parent_dot_2 = Circle(radius=0.25, color=RED, fill_opacity=1)
        self.place_at_grid(parent_dot_2, 'C3')
        
        targets_2 = VGroup(
            Circle(radius=0.25, color=BLUE, fill_opacity=1),
            Circle(radius=0.25, color=BLUE, fill_opacity=1),
            Circle(radius=0.25, color=BLUE, fill_opacity=1)
        )
        self.place_at_grid(targets_2[0], 'E2')
        self.place_at_grid(targets_2[1], 'E3')
        self.place_at_grid(targets_2[2], 'E4')
        
        dieout_label = Text("R0 < 1: Disease Dies Out", font_size=24, color=GREEN)
        # Fix Issue 52: Wider area for label
        self.place_in_area(dieout_label, 'F1', 'F6', scale_factor=0.8)
        
        self.play(FadeIn(parent_dot_2), FadeIn(targets_2))
        self.wait(0.5)
        # Recovery animation (turns green)
        self.play(parent_dot_2.animate.set_color(GREEN))
        self.play(FadeIn(dieout_label))
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(WHITE_COLOR)
        
        # Highlight the concept of "Tipping Point"
        tipping_box = SurroundingRectangle(formula, color=YELLOW_COLOR, buff=0.2)
        self.play(Create(tipping_box))
        self.wait(2)
