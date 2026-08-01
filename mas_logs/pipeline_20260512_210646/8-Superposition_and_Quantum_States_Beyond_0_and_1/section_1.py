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
            'Classical bits are like switches, either ON or OFF.', 
            'Quantum systems represent information as state vectors.', 
            'Superposition allows states to exist in a mix.'
        ]
        self.setup_layout("The Classical vs. Quantum Divide", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Classical_Switch: Draw a simple rectangle (#FFFFFF) with a circular toggle; 
        # color the toggle #FF0000 when 'OFF' (0) and #00FF00 when 'ON' (1).
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        
        switch_housing = Rectangle(width=3.0, height=1.5, color=WHITE)
        # Fix Issue 29: Move housing to area B3-D6
        self.place_in_area(switch_housing, "B3", "D6")
        
        label_off = Text("0 (OFF)", font_size=20, color=WHITE)
        label_on = Text("1 (ON)", font_size=20, color=WHITE)
        # Fix Issue 27: Move label_off to A3
        self.place_at_grid(label_off, "A3", scale_factor=0.7)
        # Fix Issue 28: Move label_on to A6
        self.place_at_grid(label_on, "A6", scale_factor=0.7)
        
        toggle = Circle(radius=0.4, color="#FF0000", fill_opacity=1.0)
        # Adjusted toggle start position to match new housing (C3)
        self.place_at_grid(toggle, "C3")
        
        self.play(
            Create(switch_housing),
            FadeIn(label_off),
            FadeIn(label_on),
            FadeIn(toggle)
        )
        self.wait(1)
        
        # Switch to ON (C6)
        self.play(
            toggle.animate.move_to(self.grid["C6"]).set_color("#00FF00"),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Quantum_State: Remove the switch and draw a vertical vector arrow (#00BFFF) 
        # centered on the screen (right side), representing a quantum bit.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#00BFFF")
        )
        
        # Positioned centered on the right side area
        start_point = self.grid["E4"]
        end_point = self.grid["B4"]
        quantum_vector = Arrow(start=start_point, end=end_point, color="#00BFFF", buff=0)
        
        self.play(
            FadeOut(switch_housing),
            FadeOut(label_off),
            FadeOut(label_on),
            FadeOut(toggle),
            GrowArrow(quantum_vector)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Superposition_Hint: The arrow smoothly rotates to a diagonal position (45 degrees) 
        # and changes color to #FFFF00, signaling a mix of states.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FFFF00")
        )
        
        # Rotation to diagonal (45 degrees clockwise looking at grid, which is -PI/4)
        self.play(
            Rotate(quantum_vector, angle=-PI/4, about_point=quantum_vector.get_start()),
            quantum_vector.animate.set_color("#FFFF00"),
            run_time=1.5
        )
        
        # Final display state
        self.wait(2)
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
