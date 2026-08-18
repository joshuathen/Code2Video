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
        lecture_lines = [
            "Diffraction is waves bending around barriers.",
            "Light spreads out passing through small apertures.",
            "This confirms light's wave-like nature."
        ]
        self.setup_layout("Diffraction: Bending Around Barriers", lecture_lines)
        
        # Load SVGs
        aperture = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/aperture.svg")
        barrier = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/barrier.svg")
        
        # Define mobjects
        ap1 = aperture.copy()
        ap2 = aperture.copy()
        self.place_at_grid(ap1, 'C2', scale_factor=0.6)
        self.place_at_grid(ap2, 'C5', scale_factor=0.6)
        
        waves = VGroup(
            *[Circle(radius=r, color=WHITE, stroke_width=2) for r in [0.5, 1.0, 1.5]]
        )
        wave1 = waves.copy()
        wave2 = waves.copy()
        wave1.move_to(ap1.get_center())
        wave2.move_to(ap2.get_center())

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.play(FadeIn(ap1), FadeIn(ap2), Create(wave1), Create(wave2))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFD700"))
        # Add barrier
        bar = barrier.copy()
        self.place_at_grid(bar, 'C3', scale_factor=0.5)
        self.play(FadeIn(bar))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        
        # Color constructive areas #00FF00 (bright green) and destructive at #FF0000 (red)
        constructive = Dot(self.grid["C3"], color="#00FF00")
        destructive = Dot(self.grid["B3"], color="#FF0000")
        
        self.play(FadeIn(constructive), FadeIn(destructive))
        self.wait(2)
