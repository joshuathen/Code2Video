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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Complex signals contain many hidden individual frequencies.",
            "Think of a prism splitting white light.",
            "Fourier transforms act like a mathematical prism.",
            "They decompose signals into their constituent parts.",
            "Everything is built from simple wave pieces."
        ]
        self.setup_layout("The Intuitive Hook: The Prism Analogy", lecture_lines)
        
        # Define visual elements
        beam = Line(start=self.grid["C1"], end=self.grid["C3"], color=WHITE, stroke_width=6)
        
        # Load asset
        prism = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/prism.svg")
        self.place_in_area(prism, 'B2', 'D4', scale_factor=0.9)
        
        rays = VGroup(
            Line(self.grid["C3"], self.grid["B6"], color=RED, stroke_width=4),
            Line(self.grid["C3"], self.grid["C6"], color=GREEN, stroke_width=4),
            Line(self.grid["C3"], self.grid["D6"], color=BLUE, stroke_width=4)
        )
        
        label_white = Text("White Light", font_size=20, color=WHITE)
        self.place_at_grid(label_white, 'B3', scale_factor=0.8)
        
        label_rainbow = Text("Rainbow", font_size=20, color=YELLOW)
        self.place_at_grid(label_rainbow, 'D5', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.play(Create(beam), Write(label_white))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(GREEN))
        self.play(FadeIn(prism))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(YELLOW))
        self.play(Create(rays), Write(label_rainbow))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[2].animate.set_color(WHITE), self.lecture[3].animate.set_color(BLUE))
        self.play(rays.animate.set_stroke(width=6))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[3].animate.set_color(WHITE), self.lecture[4].animate.set_color(RED))
        self.wait(1)
