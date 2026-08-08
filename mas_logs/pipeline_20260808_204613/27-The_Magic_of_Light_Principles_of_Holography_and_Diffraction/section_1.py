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
        lecture_lines = [
            "Light acts as an electromagnetic wave.",
            "Waves possess both amplitude and phase.",
            "Superposition creates interference patterns.",
            "Constructive interference adds peak to peak.",
            "Destructive interference cancels peaks out."
        ]
        self.setup_layout("Prerequisite: The Wave Nature of Light", lecture_lines)
        
        # Load Assets
        bulb = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bulb.svg")
        barrier = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/barrier.svg")
        screen = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/screen.svg")
        laser = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg")
        
        # Animations
        # === Animation for Lecture Line 1 ===
        self.place_at_grid(bulb, "A2", scale_factor=0.3)
        self.play(FadeIn(bulb), self.lecture[0].animate.set_color(BLUE))

        # === Animation for Lecture Line 2 ===
        # Represent wave propagation patterns with barrier
        self.place_at_grid(barrier, "B2", scale_factor=0.3)
        self.play(FadeIn(barrier), self.lecture[1].animate.set_color(YELLOW))

        # === Animation for Lecture Line 3 ===
        # Interference fringes on screen
        self.place_at_grid(screen, "C5", scale_factor=0.3)
        self.play(FadeIn(screen), self.lecture[2].animate.set_color(RED))

        # === Animation for Lecture Line 4 ===
        # Coherent frequency color change for bulb
        self.play(bulb.animate.set_color(GREEN), self.lecture[3].animate.set_color(GREEN))

        # === Animation for Lecture Line 5 ===
        # Laser morph
        self.place_at_grid(laser, "E5", scale_factor=0.3)
        self.play(FadeIn(laser), self.lecture[4].animate.set_color(PURPLE))
        
        self.wait(2)
