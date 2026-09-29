from manim import *
import os

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
        self.setup_layout("Synthesis and Summary", [
            "- Hashing and cryptography secure the ledger.",
            "- Math acts as the network's unbiased judge.",
            "- We achieve trust without human intermediaries."
        ])
        
        # Define elements
        # Using placeholder Mobjects per existing patterns
        chart = Square(color=BLUE).scale(0.5)
        network = Circle(color=RED).scale(0.5)
        ledger_asset = Square(color=PURPLE).scale(0.5) # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/ledger.svg
        trust_logo = RegularPolygon(n=5, color=YELLOW).scale(0.5)
        anchor = Dot(color=GREEN).scale(2)
        
        # === Animation for Lecture Line 1 ===
        self.place_at_grid(chart, 'B4', scale_factor=0.5)
        self.play(FadeIn(chart))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        self.place_at_grid(network, 'D4', scale_factor=0.5)
        self.place_at_grid(ledger_asset, 'D6', scale_factor=0.4)
        
        # Animate network pulses connecting the ledger asset
        pulse = Dot(color=RED).move_to(network.get_center())
        self.play(FadeIn(network), FadeIn(ledger_asset))
        self.play(pulse.animate.move_to(ledger_asset.get_center()), run_time=1)
        self.remove(pulse)
        
        self.lecture[1].set_color(YELLOW)
        
        # === Animation for Lecture Line 3 ===
        self.place_at_grid(trust_logo, 'F2', scale_factor=0.5)
        self.place_at_grid(anchor, 'F6', scale_factor=1)
        self.play(FadeIn(trust_logo), FadeIn(anchor))
        self.lecture[2].set_color(YELLOW)
        self.wait(2)
