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
        lecture_lines = [
            "The CLT allows reliable population inferences.",
            "We use small samples to estimate reality.",
            "It guarantees accuracy in mass quality control.",
            "Confidence intervals predict large-scale outcomes.",
            "Statistics makes massive data manageable."
        ]
        self.setup_layout("Real-World Application", lecture_lines)
        
        # Elements
        text_clt = Text("Use CLT for reliable population estimates.", color=WHITE, font_size=24)
        sample_bar = Rectangle(height=0.5, width=2.0, fill_color=YELLOW, fill_opacity=0.8, stroke_width=0)
        conf_range = Rectangle(height=0.5, width=4.0, fill_color=BLUE_E, fill_opacity=0.3, stroke_width=2, stroke_color=YELLOW)
        
        # Assets
        clipboard = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/clipboard.svg")
        scanner = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/scanner.svg")
        
        # Initial positions
        self.place_at_grid(text_clt, 'C2', scale_factor=0.7)
        self.place_at_grid(clipboard, 'B2', scale_factor=0.5)
        self.place_at_grid(sample_bar, 'D2', scale_factor=0.6)
        self.place_at_grid(conf_range, 'D4', scale_factor=0.6)
        self.place_at_grid(scanner, 'E4', scale_factor=0.5)
        
        # Hide assets initially
        clipboard.set_opacity(0)
        scanner.set_opacity(0)
        sample_bar.set_opacity(0)
        conf_range.set_opacity(0)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.play(FadeIn(text_clt), clipboard.animate.set_opacity(1))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.play(FadeIn(sample_bar))
        self.play(sample_bar.animate.move_to(self.grid['D4']))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(WHITE))
        self.play(FadeIn(conf_range), scanner.animate.set_opacity(1))
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.play(conf_range.animate.set_stroke(width=8, color=WHITE))
        self.wait(2)
