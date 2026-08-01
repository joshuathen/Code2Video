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
        lecture_lines = [
            'Eigenvalues describe the fundamental scaling power of transformations.', 
            "They power Google's PageRank and analyze bridge vibrations.", 
            'They are the DNA of every linear transformation.'
        ]
        self.setup_layout("Summary & Application", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"), run_time=0.5)
        
        # Display text 'Eigenvalues = Scaling Power and Stability' in yellow (#FFFF00)
        scaling_line1 = Text("Eigenvalues =", color="#FFFF00", font_size=24)
        scaling_line2 = Text("Scaling Power", color="#FFFF00", font_size=24)
        scaling_line3 = Text("& Stability", color="#FFFF00", font_size=24)
        scaling_text_group = VGroup(scaling_line1, scaling_line2, scaling_line3).arrange(DOWN, buff=0.3)
        
        self.place_in_area(scaling_text_group, "A1", "C6")
        self.play(FadeIn(scaling_text_group))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FFFF00"),
            run_time=0.5
        )
        
        # Show icons for 'Google PageRank' [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/server.svg] 
        # and 'Bridge Vibrations' [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/bridge.svg] in white (#FFFFFF)
        google_label = Text("Google PageRank", font_size=18, color=WHITE)
        google_svg = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/server.svg").set_color(WHITE).scale(0.5)
        google_icon = VGroup(google_svg, google_label).arrange(DOWN, buff=0.2)
        
        bridge_label = Text("Bridge Vibrations", font_size=18, color=WHITE)
        bridge_svg = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/bridge.svg").set_color(WHITE).scale(0.5)
        bridge_icon = VGroup(bridge_svg, bridge_label).arrange(DOWN, buff=0.2)
        
        # Fix: Issue 42 - Adjusted positioning for google_icon
        self.place_in_area(google_icon, 'D1', 'F3', scale_factor=0.8)
        # Fix: Issue 43 - Adjusted positioning for bridge_icon
        self.place_in_area(bridge_icon, 'D4', 'F6', scale_factor=0.8)
        
        self.play(FadeIn(google_icon), FadeIn(bridge_icon))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FFFF00"),
            run_time=0.5
        )
        
        # Fade to black (clear the right-side elements)
        self.play(
            FadeOut(scaling_text_group),
            FadeOut(google_icon),
            FadeOut(bridge_icon)
        )
        
        # Center a single gold (λ) symbol (#FFD700)
        dna_lambda = Text("λ", color="#FFD700", font_size=150)
        # Fix: Issue 41 - Adjusted λ positioning and area
        self.place_in_area(dna_lambda, 'B2', 'E5', scale_factor=1.0)
        
        self.play(FadeIn(dna_lambda))
        self.wait(3)
