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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Duality Challenge", [
            "Signals narrow in time broaden in frequency.",
            "A brief pulse requires infinite frequencies.",
            "This is the core duality challenge."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Display a single point particle (#33A1FF) on screen [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/pulse.svg].
        particle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pulse.svg")
        particle.set_color("#33A1FF")
        self.place_at_grid(particle, 'A4', scale_factor=0.7)
        self.play(FadeIn(particle))
        self.lecture[0].set_color("#33A1FF")

        # === Animation for Lecture Line 2 ===
        # Expand the point particle into a wave packet (#A133FF).
        wave_packet = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pulse.svg")
        wave_packet.set_color("#A133FF")
        self.place_at_grid(wave_packet, 'A6', scale_factor=0.7)
        self.play(ReplacementTransform(particle.copy(), wave_packet))
        self.lecture[1].set_color("#A133FF")

        # === Animation for Lecture Line 3 ===
        # Show the trade-off by shrinking spatial width while frequency spreads [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/pulse.svg].
        trade_off_viz = VGroup(
            Line(LEFT, RIGHT, color=WHITE),
            Dot(color="#33A1FF"),
            Dot(color="#A133FF")
        )
        self.place_in_area(trade_off_viz, 'D2', 'F5', scale_factor=0.5)
        self.play(FadeIn(trade_off_viz))
        self.lecture[2].set_color(YELLOW)
        self.wait(2)
