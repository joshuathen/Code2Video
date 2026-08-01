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
        # Initial Setup
        title_text = "The Classical vs. Quantum Divide"
        lecture_lines = [
            "Classical states are binary, like a simple toggle switch.",
            "Quantum states resemble a spinning, blurred coin.",
            "This coexistence of states defines quantum superposition."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Matching color: White
        self.lecture[0].set_color("#FFFFFF")
        
        # Create Switch Graphic using Asset
        switch_svg = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/switch.svg").set_color("#FFFFFF")
        self.place_in_area(switch_svg, "A1", "C3", scale_factor=0.8)
        
        # Auxiliary labels for '0' and '1' positions
        label_0 = Text("0", font_size=20, color="#FFFFFF")
        label_1 = Text("1", font_size=20, color="#FFFFFF")
        self.place_at_grid(label_0, "B1", scale_factor=0.8)
        self.place_at_grid(label_1, "B3", scale_factor=0.8)
        
        self.play(FadeIn(switch_svg), FadeIn(label_0), FadeIn(label_1))
        self.wait(0.5)
        
        # Simulate click between states
        self.play(switch_svg.animate.flip(axis=UP), run_time=0.4)
        self.play(switch_svg.animate.flip(axis=UP), run_time=0.3)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Matching color: Cyan
        self.lecture[1].set_color("#00FFFF")
        
        # Create spinning coin graphic using Asset
        coin_svg = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/coin.svg").set_color("#00FFFF")
        self.place_in_area(coin_svg, "A4", "C6", scale_factor=0.8)
        
        self.play(FadeIn(coin_svg))
        
        # Rotate rapidly to simulate blur
        self.play(
            Rotate(coin_svg, axis=UP, angle=TAU * 10, run_time=3, rate_func=linear)
        )
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Matching color: Yellow
        self.lecture[2].set_color("#FFFF00")
        
        classic_label = Text("Classical: Binary", font_size=24, color="#FFFF00")
        quantum_label = Text("Quantum: Superposition", font_size=24, color="#FFFF00")
        
        # Fixed positioning and scaling per VideoCritic feedback
        self.place_in_area(classic_label, 'D1', 'D3', scale_factor=0.7)
        self.place_in_area(quantum_label, 'D4', 'D6', scale_factor=0.7)

        self.play(Write(classic_label), Write(quantum_label))
        self.wait(2)
