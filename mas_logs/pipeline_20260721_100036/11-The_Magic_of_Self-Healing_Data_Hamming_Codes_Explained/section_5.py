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
        title = "Real-World Application: ECC Memory"
        lines = [
            "Hamming codes power Error Correction Code (ECC) memory.",
            "They protect servers and deep-space communication from corruption.",
            "Self-healing data keeps our digital world running smoothly."
        ]
        self.setup_layout(title, lines)

        # RAM Stick Asset
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/ram.svg]
        ram_stick = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ram.svg")
        ram_stick.set_color("#A9A9A9")
        
        # === Animation for Lecture Line 1 ===
        # Hamming codes power Error Correction Code (ECC) memory.
        self.lecture[0].set_color(YELLOW)
        # Positioned to avoid overlapping text as per Issue 31
        self.place_in_area(ram_stick, "B3", "E6", scale_factor=0.8)
        self.play(FadeIn(ram_stick))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # They protect servers and deep-space communication from corruption.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#00FFFF") # Cyan

        # Create cosmic rays (cyan particles)
        particles = VGroup(*[
            Dot(radius=0.06, color="#00FFFF")
            for _ in range(12)
        ])
        
        # Position particles to originate from the top-right
        for p in particles:
            start_pos = self.grid["A6"] + np.array([np.random.uniform(0, 2), np.random.uniform(0, 2), 0])
            p.move_to(start_pos)

        self.play(FadeIn(particles))
        
        # Animate particles hitting the RAM and fading out
        particle_anims = []
        for p in particles:
            # Target random points on the RAM stick area
            target = ram_stick.get_center() + np.array([np.random.uniform(-1.0, 1.0), np.random.uniform(-0.4, 0.4), 0])
            particle_anims.append(p.animate.move_to(target).set_opacity(0))
        
        self.play(*particle_anims, run_time=1.5, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Self-healing data keeps our digital world running smoothly.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#00FF00") # Green

        # Bit "healing" animation
        # Place a small square on the RAM to represent a bit
        bit = Square(side_length=0.15, fill_opacity=1, stroke_width=0)
        bit.move_to(ram_stick.get_center() + RIGHT * 0.5)
        bit.set_color(RED)

        # Show bit turning red (corruption)
        self.play(FadeIn(bit))
        self.play(Flash(bit, color=RED, flash_radius=0.3))
        self.wait(0.5)

        # Correction animation: Red to Green
        self.play(bit.animate.set_color("#00FF00"))
        self.play(Indicate(bit, color="#00FF00", scale_factor=1.5))
        
        # Final fade
        self.play(FadeOut(bit), run_time=1)
        
        self.wait(2)
