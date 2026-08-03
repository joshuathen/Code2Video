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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisite: The Conservation Laws", [
            "Two physical laws govern every single bounce.",
            "Kinetic energy and momentum are always conserved.",
            "These laws define how velocities change over time."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Initial highlight
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second line
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Display the Energy Conservation equation in magenta.
        # Fix Issue 28: Move energy_eq to A2-B5
        energy_eq = MathTex(
            "\\frac{1}{2} M v_1^2 + \\frac{1}{2} m v_2^2 = E",
            color="#FF00FF"
        )
        self.place_in_area(energy_eq, "A2", "B5", scale_factor=0.8)
        
        # Display the Momentum Conservation equation in cyan.
        # Fix Issue 29: Move momentum_eq to C2-D5
        momentum_eq = MathTex(
            "M v_1 + m v_2 = P",
            color="#00FFFF"
        )
        self.place_in_area(momentum_eq, "C2", "D5", scale_factor=0.8)

        self.play(Write(energy_eq))
        self.wait(0.5)
        self.play(Write(momentum_eq))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Highlight third line
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )

        # Show a state point representing velocities changing per collision.
        # Fix Issue 25: Integrate asset bounce.svg
        # Fix Issue 30: Move state_point to E3
        state_point = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bounce.svg")
        self.place_at_grid(state_point, "E3", scale_factor=0.3)
        state_point.set_color(WHITE)
        
        # Initial label
        state_label = MathTex("(v_1, v_2)", font_size=24, color=WHITE)
        state_label.next_to(state_point, UP, buff=0.1)
        
        self.play(FadeIn(state_point), FadeIn(state_label))
        self.wait(1)
        
        # Target position and label for the "change"
        # Moving to E5 to keep it on the same row as E3 (Fix for Issue 30 consistency)
        new_pos = self.grid["E5"]
        new_label = MathTex("(v_1', v_2')", font_size=24, color=WHITE)
        new_label.move_to(new_pos + UP * 0.4)
        
        # Move point and transform label
        self.play(
            state_point.animate.move_to(new_pos),
            Transform(state_label, new_label)
        )
        self.wait(2)
        
        # Final cleanup: fade lecture highlight
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(2)
