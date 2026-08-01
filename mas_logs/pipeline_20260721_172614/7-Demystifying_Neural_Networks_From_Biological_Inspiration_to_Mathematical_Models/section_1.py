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
        self.setup_layout("The Biological Blueprint", [
            "Nature inspired the design of artificial neural networks.",
            "Dendrites receive signals and send them to the cell body.",
            "The axon then transmits the processed signal forward."
        ])

        # Colors
        neuron_color = "#ADD8E6"
        pulse_color = "#FFFFE0"
        flash_color = "#FFFFFF"
        digital_color = "#00FF00"

        # === Animation for Lecture Line 1 ===
        # Draw a stylized biological neuron [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/neuron.svg] (#ADD8E6) with labeled parts: Dendrites, Cell Body, Axon.
        self.lecture[0].set_color(YELLOW)
        
        # Load and place asset (Issue 19)
        neuron_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/neuron.svg")
        neuron_svg.set_color(neuron_color)
        # Position neuron centrally on the right side
        self.place_in_area(neuron_svg, 'B1', 'D5', scale_factor=1.5)
        neuron_center = neuron_svg.get_center()
        
        # Labels
        label_d = Text("Dendrites", font_size=18, color=neuron_color)
        self.place_at_grid(label_d, 'B1', scale_factor=0.8) # Fix: Issue 22
        
        label_cb = Text("Cell Body", font_size=18, color=neuron_color)
        self.place_at_grid(label_cb, 'B3', scale_factor=0.8) # Adjusted for center of SVG
        
        label_a = Text("Axon", font_size=18, color=neuron_color)
        self.place_at_grid(label_a, 'B5', scale_factor=0.8)

        labels_vgroup = VGroup(label_d, label_cb, label_a)

        self.play(DrawBorderThenFill(neuron_svg), Write(labels_vgroup), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Animate small pulses (#FFFFE0) moving from Dendrites into the Cell Body.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # Define paths for incoming pulses
        pulse_start_points = [self.grid['B1'], self.grid['C1'], self.grid['D1']]
        pulses = VGroup(*[Dot(radius=0.08, color=pulse_color) for _ in range(6)])
        paths = [Line(start, neuron_center) for start in pulse_start_points]

        self.play(
            Succession(
                MoveAlongPath(pulses[0], paths[0], run_time=0.8),
                MoveAlongPath(pulses[1], paths[1], run_time=0.8),
                MoveAlongPath(pulses[2], paths[2], run_time=0.8),
            ),
            Succession(
                Wait(0.4),
                MoveAlongPath(pulses[3], paths[0], run_time=0.8),
                MoveAlongPath(pulses[4], paths[1], run_time=0.8),
                MoveAlongPath(pulses[5], paths[2], run_time=0.8),
            )
        )
        
        # Cell Body flashes brightly (#FFFFFF) as pulses gather inside.
        flash_circle = Circle(radius=0.4, color=flash_color, fill_opacity=0.8).move_to(neuron_center)
        
        self.play(
            FadeIn(flash_circle),
            *[FadeOut(p) for p in pulses],
            run_time=0.3
        )
        self.play(FadeOut(flash_circle), run_time=0.3)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # A single large pulse (#FFFFE0) travels quickly down the Axon.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        large_pulse = Dot(radius=0.15, color=pulse_color)
        # Axon path from center toward the right side
        axon_path = Line(neuron_center, self.grid['C6'])
        
        self.play(MoveAlongPath(large_pulse, axon_path), run_time=1.2, rate_func=exponential_decay)
        self.play(FadeOut(large_pulse))
        self.wait(1)

        # Transition the biological neuron into a digital node (#00FF00) symbol.
        digital_node = Circle(radius=0.7, color=digital_color, stroke_width=6)
        self.place_in_area(digital_node, 'B3', 'D5', scale_factor=1.0) # Fix: Issue 23
        
        sigma = MathTex(r"\Sigma", color=digital_color, font_size=40)
        self.place_in_area(sigma, 'B3', 'D5', scale_factor=1.0) # Fix: Issue 24
        
        digital_vgroup = VGroup(digital_node, sigma)

        self.play(
            FadeOut(neuron_svg),
            FadeOut(labels_vgroup),
            FadeIn(digital_vgroup),
            run_time=2
        )
        self.wait(2)
