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

class Section5Scene(TeachingScene):
    def construct(self):
        # Initialize Scene with title and lecture lines from snapshot
        self.setup_layout(
            "Modern Understanding: An Infinite Mystery", 
            [
                "Pi is irrational, so its digits never end.",
                "These digits stretch infinitely into the stars.",
                "The symbol pi represents this eternal mathematical bridge."
            ]
        )
        
        # Colors for lecture lines
        line_colors = ["#00FFFF", "#ADD8E6", "#FFD700"]

        # === Animation for Lecture Line 1 ===
        # Pi is irrational, so its digits never end.
        self.play(self.lecture[0].animate.set_color(line_colors[0]))
        
        # Create a long stream of Pi digits
        pi_str = "3.141592653589793238462643383279502884197169399375105820974944592307816406286..."
        digit_stream = Text(pi_str, font_size=36, color=line_colors[0])
        
        # Initial placement as requested
        self.place_in_area(digit_stream, 'B4', 'B6', scale_factor=0.4)
        digit_stream.shift(RIGHT * 6) # Start further off-screen to the right
        
        self.add(digit_stream)
        
        # Animate stream flowing rapidly across the screen
        self.play(
            digit_stream.animate.shift(LEFT * 14),
            run_time=4,
            rate_func=linear
        )
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # These digits stretch infinitely into the stars.
        self.play(self.lecture[1].animate.set_color(line_colors[1]))
        
        # Load starry background asset
        stars_asset = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/stars.svg")
        self.place_in_area(stars_asset, "A1", "F6", scale_factor=2.0)
        stars_asset.set_z_index(-1) # Ensure stars are in background
        stars_asset.set_opacity(0.6)

        # Transition: Background to deep blue and digits recede
        # Fix: camera.animate does not exist in Manim CE, change background_color directly
        self.camera.background_color = "#000033"
        self.play(
            FadeIn(stars_asset),
            digit_stream.animate.scale(0.1).move_to(self.grid["C3"]).set_opacity(0.3),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The symbol pi represents this eternal mathematical bridge.
        self.play(self.lecture[2].animate.set_color(line_colors[2]))
        
        # Golden pi symbol
        pi_symbol = Text("π", color="#FFD700")
        self.place_at_grid(pi_symbol, 'D4', scale_factor=1.2)
        
        # Digits vanish as Pi symbol pulses
        self.play(
            FadeOut(digit_stream),
            FadeIn(pi_symbol),
            run_time=1
        )
        
        # Pulsing animation for Pi symbol
        for _ in range(2):
            self.play(
                pi_symbol.animate.scale(1.2),
                rate_func=there_and_back,
                run_time=1.5
            )

        self.wait(2)
