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
        title_text = "Defining Superposition: The Spinning Coin"
        lecture_lines = [
            "Quantum systems can exist in multiple states at once.",
            "Imagine a spinning coin, neither heads nor tails.",
            "This \"blur\" of possibilities is known as quantum superposition."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Fade in 'Superposition' in bright blue (#0000FF) at the top.
        self.play(self.lecture[0].animate.set_color("#0000FF"))
        
        superposition_text = Text("Superposition", color="#0000FF", font_size=36)
        self.place_in_area(superposition_text, 'A2', 'A5')
        self.play(Write(superposition_text))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Display a silver circle (#C0C0C0) representing a coin in the center.
        # Animate the coin spinning rapidly using elliptical scaling.
        self.play(self.lecture[1].animate.set_color("#C0C0C0"))
        
        coin = Circle(radius=1.5, color="#C0C0C0", fill_opacity=0.5)
        self.place_in_area(coin, 'B3', 'E4')
        
        # To simulate spinning, we'll use a ValueTracker and an updater.
        spin_tracker = ValueTracker(0)
        # Using stretch_to_fit_width to simulate rotation around Y axis. 
        # Initial radius is 1.5, so width is 3.0.
        coin.add_updater(lambda m: m.stretch_to_fit_width(
            3.0 * max(0.05, abs(np.cos(spin_tracker.get_value() * PI)))
        ))
        
        self.play(Create(coin))
        self.play(spin_tracker.animate.set_value(5), run_time=3, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Show labels '|Heads⟩' (#00FF00) and '|Tails⟩' (#FF00FF) appearing over the blur.
        # Pulse the blurred coin to emphasize the coexistence of both states.
        self.play(self.lecture[2].animate.set_color("#00FF00")) 
        
        heads_label = MathTex(r"|\text{Heads}\rangle", color="#00FF00")
        tails_label = MathTex(r"|\text{Tails}\rangle", color="#FF00FF")
        
        # Applying requested fixes for issues 22 and 23 to avoid overlap with coin
        self.place_at_grid(heads_label, 'C2')
        self.place_at_grid(tails_label, 'D5')
        
        # Pulse animation
        pulse_anim = coin.animate(rate_func=there_and_back).scale(1.2)
        
        self.play(
            FadeIn(heads_label),
            FadeIn(tails_label),
            pulse_anim,
            run_time=2
        )
        self.wait(2)
