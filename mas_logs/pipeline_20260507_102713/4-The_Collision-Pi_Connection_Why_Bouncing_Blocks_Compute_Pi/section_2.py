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
        # Section setup
        title_text = "Prerequisite: The Laws of the Game"
        lecture_lines = [
            'Momentum is conserved as the blocks push each other.', 
            'Kinetic energy is also preserved in these elastic bounces.', 
            'These two laws define the outcome of every collision.'
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Initial Dimming
        for line in self.lecture:
            line.set_color(GRAY)

        # === Animation for Lecture Line 1 ===
        # Line: "Momentum is conserved as the blocks push each other."
        # Display the Conservation of Momentum equation: 'm v_1 + M v_2 = const' in yellow (#FFFF00)
        # Accompanied by the blocks icon [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/blocks.svg]
        momentum_eq = Text("m v1 + M v2 = const", color="#FFFF00")
        self.place_in_area(momentum_eq, 'B1', 'C6', scale_factor=0.9)
        
        blocks_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/blocks.svg")
        self.place_at_grid(blocks_icon, 'A4', scale_factor=0.6)
        
        self.play(
            self.lecture[0].animate.set_color("#FFFF00"),
            Write(momentum_eq),
            FadeIn(blocks_icon),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line: "Kinetic energy is also preserved in these elastic bounces."
        # Display the Conservation of Energy equation: '1/2 m v1^2 + 1/2 M v2^2 = E' in cyan (#00FFFF)
        energy_eq = Text("1/2 m v1^2 + 1/2 M v2^2 = E", color="#00FFFF")
        self.place_in_area(energy_eq, 'D1', 'E6', scale_factor=0.8)
        
        self.play(
            self.lecture[1].animate.set_color("#00FFFF"),
            Write(energy_eq),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line: "These two laws define the outcome of every collision."
        # Highlight both equations with a white rectangle (#FFFFFF) to indicate they are governing laws.
        highlight_box = SurroundingRectangle(VGroup(momentum_eq, energy_eq), color=WHITE, buff=0.2)
        
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            Create(highlight_box),
            run_time=1.5
        )
        self.wait(2)
