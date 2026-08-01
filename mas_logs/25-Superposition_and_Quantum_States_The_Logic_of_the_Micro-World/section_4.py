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

class Section4Scene(TeachingScene):
    def construct(self):
        # Setup the layout
        title = "The Act of Measurement (Wavefunction Collapse)"
        lines = [
            'Consider a vector in a specific superposition state.',
            'Observing the system triggers a process called measurement.',
            'The system fluctuates between potential classical outcomes.',
            'Measurement forces the wavefunction to collapse into one state.',
            'The probability of this outcome is the amplitude squared.'
        ]
        self.setup_layout(title, lines)

        # Helper for coordinates
        origin = self.grid['D2']
        x_end = self.grid['D5']
        y_end = self.grid['A2']
        
        # === Animation for Lecture Line 1 ===
        # Consider a vector in a specific superposition state.
        self.lecture[0].set_color("#00FFFF")
        
        # Create Axes
        h_axis = Arrow(origin, x_end, buff=0, color=GRAY_B)
        v_axis = Arrow(origin, y_end, buff=0, color=GRAY_B)
        label_0 = Text("|0⟩", font_size=24).next_to(x_end, RIGHT, buff=0.1)
        label_1 = Text("|1⟩", font_size=24).next_to(y_end, UP, buff=0.1)
        
        # Create Vector |ψ⟩ at 30 degrees
        vec_len = 2.5
        angle = 30 * DEGREES
        tip_pos = origin + np.array([vec_len * np.cos(angle), vec_len * np.sin(angle), 0])
        psi_vec = Arrow(origin, tip_pos, buff=0, color="#00FFFF")
        psi_label = Text("|ψ⟩", font_size=24, color="#00FFFF").next_to(tip_pos, UR, buff=0.1)
        
        self.play(Create(h_axis), Create(v_axis), FadeIn(label_0), FadeIn(label_1))
        self.play(GrowArrow(psi_vec), FadeIn(psi_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Observing the system triggers a process called measurement.
        self.lecture[1].set_color(WHITE)
        
        # Camera Icon [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/camera.svg]
        # Issue 41: Move to B5, scale 0.8
        camera = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/camera.svg")
        camera.set_color(WHITE)
        self.place_at_grid(camera, 'B5', scale_factor=0.8)
        
        self.play(FadeIn(camera))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The system fluctuates between potential classical outcomes.
        self.lecture[2].set_color(WHITE)
        
        # Flickering effect between |0> and |1>
        # We simulate fluctuation by rotating slightly towards axes
        tip_0 = origin + np.array([vec_len, 0, 0])
        tip_1 = origin + np.array([0, vec_len, 0])
        
        for _ in range(3):
            self.play(psi_vec.animate.rotate(10*DEGREES, about_point=origin), run_time=0.15)
            self.play(psi_vec.animate.rotate(-20*DEGREES, about_point=origin), run_time=0.15)
            self.play(psi_vec.animate.rotate(10*DEGREES, about_point=origin), run_time=0.15)
            
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        # Measurement forces the wavefunction to collapse into one state.
        self.lecture[3].set_color("#00FF00")
        
        # Snap vector to |0⟩ axis
        collapsed_tip = origin + np.array([vec_len, 0, 0])
        
        self.play(
            ReplacementTransform(psi_vec, Arrow(origin, collapsed_tip, buff=0, color="#00FF00")),
            psi_label.animate.next_to(collapsed_tip, UP, buff=0.1).set_color("#00FF00"),
            camera.animate.scale(1.2).set_opacity(0.5), # Pulse camera
            run_time=0.5
        )
        self.play(FadeOut(camera), run_time=0.3)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The probability of this outcome is the amplitude squared.
        self.lecture[4].set_color("#FFFF00")
        
        outcome_text = Text("Outcome: |0⟩", font_size=24, color="#FFFF00")
        prob_text = Text("Prob = |α|²", font_size=24, color="#FFFF00")
        labels_vgroup = VGroup(outcome_text, prob_text).arrange(DOWN, aligned_edge=LEFT)
        
        # Issue 42: Place in area B6 to C6, scale 0.8
        self.place_in_area(labels_vgroup, 'B6', 'C6', scale_factor=0.8)
        
        self.play(Write(labels_vgroup))
        self.wait(2)
