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
        self.setup_layout("The Cosmic Noise Problem", [
            "Data travels across space as sequences of binary bits.",
            "Cosmic rays can flip a bit from zero to one.",
            "One flipped bit can corrupt an entire digital file."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Match color with Voyager/Earth (Blue-ish/White)
        self.play(self.lecture[0].animate.set_color(BLUE_B))
        
        # Voyager-like probe (representing data source)
        voyager = VGroup(
            Circle(radius=0.5, color=GRAY),
            Line(LEFT*0.5, RIGHT*0.5, color=GRAY),
            Line(UP*0.5, DOWN*0.5, color=GRAY),
            Circle(radius=0.1, color=BLUE).shift(UP*0.3)
        )
        self.place_in_area(voyager, 'B1', 'C1', scale_factor=0.6)
        
        # Earth (representing receiver)
        earth = VGroup(
            Circle(radius=0.6, color=BLUE_E, fill_opacity=1),
            Circle(radius=0.55, color=GREEN, fill_opacity=0.3)
        )
        # Fix for Issue 28: Reposition Earth to avoid overlap with bits in row B
        self.place_in_area(earth, 'C6', 'D6', scale_factor=0.8)
        
        # Bit string '1011'
        bits_val = ["1", "0", "1", "1"]
        bits = VGroup(*[Text(b, font_size=36, color=WHITE) for b in bits_val]).arrange(RIGHT, buff=0.3)
        self.place_at_grid(bits, 'B2')
        
        self.play(Create(voyager), Create(earth))
        self.play(Write(bits))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Match color with Cosmic Ray (Red)
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(RED)
        )
        
        # Bits move halfway
        self.play(bits.animate.move_to(self.grid['B3']), run_time=1.5)
        
        # Red glowing particle (Cosmic Ray)
        particle = Dot(color=RED, radius=0.15)
        glow = Arc(radius=0.2, angle=TAU, color=RED).set_stroke(width=10, opacity=0.5)
        particle_group = VGroup(particle, glow)
        # Fix for Issue 29: Use place_in_area for particle_group
        self.place_in_area(particle_group, 'A2', 'A3', scale_factor=1.0)
        
        self.play(FadeIn(particle_group))
        
        # Strike the second bit ('0')
        # We target bits[1] which is the second character in '1011'
        self.play(particle_group.animate.move_to(bits[1].get_center()), run_time=0.6)
        
        # Bit '0' flips to '1' and turns red
        bit_zero = bits[1]
        bit_one_red = Text("1", font_size=36, color=RED).move_to(bit_zero.get_center())
        
        self.play(
            FadeOut(particle_group),
            Transform(bit_zero, bit_one_red),
            Flash(bit_zero, color=RED)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Match color with corruption state (Red)
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(RED)
        )
        
        # Bits travel to Earth
        self.play(bits.animate.move_to(self.grid['B6']), run_time=1.5)
        
        # Warning display near Earth
        warning_text = Text("DATA CORRUPTED", color=RED, font_size=24, weight=BOLD)
        # Fix for Issue 27: Reposition warning_text to avoid screen cutoff
        self.place_in_area(warning_text, 'E5', 'E6', scale_factor=0.9)
        
        self.play(
            Write(warning_text),
            earth.animate.set_color(RED_E)
        )
        self.wait(2)
