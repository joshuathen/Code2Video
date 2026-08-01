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
        # Setup layout with title and lecture lines
        title_text = "Prerequisite: The DNA of Combinatorics"
        lecture_lines = [
            'Combinatorics uses polynomials to store and count sequences.',
            'Each coefficient represents the number of ways to choose.',
            'For three apples, (1+x)^3 tracks all possible selections.'
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Script: 'Combinatorics uses polynomials to store and count sequences.'
        self.lecture[0].set_color(WHITE)
        
        poly = Text("(1+x)³", color=WHITE)
        # Fix: Ensure uniform scale for 'poly' (Issue 42/31)
        self.place_in_area(poly, 'C3', 'D4', scale_factor=1.5)
        
        self.play(Write(poly))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Script: 'Each coefficient represents the number of ways to choose.'
        self.play(self.lecture[1].animate.set_color(WHITE))
        
        expanded = VGroup(
            Text("1"), 
            Text("+"), 
            Text("3x"), 
            Text("+"), 
            Text("3x²"), 
            Text("+"), 
            Text("1x³")
        ).arrange(RIGHT, buff=0.15).set_color(WHITE)
        
        # Fix: expanded formula position (Issue 42/29)
        self.place_in_area(expanded, 'C2', 'D6', scale_factor=1.5)
        
        self.play(ReplacementTransform(poly, expanded))
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # Script: 'For three apples, (1+x)^3 tracks all possible selections.'
        highlight_color = "#00FF00"
        self.play(self.lecture[2].animate.set_color(highlight_color))
        
        # Locate the 3x² term (index 4 in the VGroup)
        target_term = expanded[4]
        highlight_box = SurroundingRectangle(target_term, color=highlight_color, buff=0.1)
        
        label = Text("3 ways to choose 2", font_size=24, color=highlight_color)
        # Fix: Align 'label' (Issue 42/30)
        self.place_in_area(label, 'B4', 'B5', scale_factor=0.8)

        # Asset Integration (Issue 42/26)
        apple_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/apple.svg"
        
        def make_apple_pair():
            apple1 = SVGMobject(apple_path, height=0.3, color=RED)
            apple2 = SVGMobject(apple_path, height=0.3, color=RED)
            return VGroup(apple1, apple2).arrange(RIGHT, buff=0.05)

        # Create three visual pairs representing the 3 ways to choose 2
        apple_pairs = VGroup(
            make_apple_pair(),
            make_apple_pair(),
            make_apple_pair()
        ).arrange(RIGHT, buff=0.3)
        
        # Position apples below the term in row E
        self.place_in_area(apple_pairs, 'E3', 'E5', scale_factor=1.0)
        
        self.play(
            Create(highlight_box),
            FadeIn(label, shift=UP * 0.2),
            FadeIn(apple_pairs, shift=UP * 0.2)
        )
        self.wait(3)
