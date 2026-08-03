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

class Section6Scene(TeachingScene):
    def construct(self):
        # Data
        title = "Summary & The Power of Abstraction"
        lecture_lines = [
            "Stripping away arrows reveals the math's true power.",
            "These rules apply to quantum mechanics and AI.",
            "If it follows the rules, it's a vector space."
        ]
        self.setup_layout(title, lecture_lines)
        
        # Colors
        COLOR_ARROW = "#FFFFFF"
        COLOR_WAVE = "#FFFFFF"
        COLOR_MATRIX = "#FFFFFF"
        COLOR_QUANTUM = "#00FFFF"
        COLOR_AI = "#FFD700"
        COLOR_VS = "#FFFFFF"
        COLOR_DIM = "#888888"

        # Initialize all lecture lines to dimmed color
        for line in self.lecture:
            line.set_color(COLOR_DIM)
        
        # === Animation for Lecture Line 1 ===
        # A montage of a white arrow (#FFFFFF), a wave, and a matrix appears.
        self.play(self.lecture[0].animate.set_color(COLOR_ARROW))
        
        arrow = Arrow(LEFT, RIGHT, color=COLOR_ARROW)
        self.place_at_grid(arrow, 'B2', scale_factor=0.6)
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/wave.svg
        wave_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wave.svg")
        wave_svg.set_color(COLOR_WAVE)
        self.place_at_grid(wave_svg, 'B4', scale_factor=0.5)
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/matrix.svg
        # Adjusted position from B6 to B5 (Issue 33)
        matrix_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/matrix.svg")
        matrix_svg.set_color(COLOR_MATRIX)
        self.place_at_grid(matrix_svg, 'B5', scale_factor=0.5)
        
        self.play(
            FadeIn(arrow),
            FadeIn(wave_svg),
            FadeIn(matrix_svg)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Icons for 'Quantum' in cyan (#00FFFF) and 'AI' in gold (#FFD700) pulse.
        self.play(
            self.lecture[0].animate.set_color(COLOR_DIM),
            self.lecture[1].animate.set_color(COLOR_QUANTUM)
        )
        
        quantum_label = Text("Quantum", font_size=24, color=COLOR_QUANTUM)
        quantum_circle = Circle(radius=0.4, color=COLOR_QUANTUM)
        quantum_icon = VGroup(quantum_circle, quantum_label).arrange(DOWN, buff=0.2)
        self.place_at_grid(quantum_icon, 'D2', scale_factor=1.0)
        
        ai_label = Text("AI", font_size=24, color=COLOR_AI)
        ai_square = Square(side_length=0.8, color=COLOR_AI)
        ai_icon = VGroup(ai_square, ai_label).arrange(DOWN, buff=0.2)
        # Adjusted position from D5 to D4 (Issue 34)
        self.place_at_grid(ai_icon, 'D4', scale_factor=1.0)
        
        self.play(FadeIn(quantum_icon), FadeIn(ai_icon))
        
        # Pulsing logic with ValueTracker (persistent mobjects)
        pulse_tracker = ValueTracker(1.0)
        orig_q_h = quantum_icon.get_height()
        orig_a_h = ai_icon.get_height()
        
        # Add updaters for pulsing
        quantum_icon.add_updater(lambda m: m.scale_to_fit_height(orig_q_h * pulse_tracker.get_value()))
        ai_icon.add_updater(lambda m: m.scale_to_fit_height(orig_a_h * pulse_tracker.get_value()))
        
        self.play(pulse_tracker.animate.set_value(1.2), run_time=0.6, rate_func=there_and_back)
        self.play(pulse_tracker.animate.set_value(1.2), run_time=0.6, rate_func=there_and_back)
        
        quantum_icon.clear_updaters()
        ai_icon.clear_updaters()
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The words 'VECTOR SPACE' in white (#FFFFFF) scale up to fill the screen.
        self.play(
            self.lecture[1].animate.set_color(COLOR_DIM),
            self.lecture[2].animate.set_color(COLOR_VS)
        )
        
        vs_text = Text("VECTOR SPACE", font_size=48, color=COLOR_VS)
        # Adjust placement and scale factor (Issue 32)
        self.place_in_area(vs_text, 'C1', 'D6', scale_factor=0.9)
        
        # We start small for the scale-up effect
        # Store current scale (0.9 in area)
        final_vs_text = vs_text.copy()
        vs_text.scale(0.1) 
        
        self.play(
            FadeOut(arrow), FadeOut(wave_svg), FadeOut(matrix_svg),
            FadeOut(quantum_icon), FadeOut(ai_icon)
        )
        
        # Scale up to the size defined by place_in_area (Issue 32)
        self.play(vs_text.animate.scale(10), run_time=2, rate_func=smooth)
        
        self.wait(3)
