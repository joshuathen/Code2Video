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
        self.setup_layout(
            "Summary and Real-World Impact",
            [
                "Networks learn by adjusting weights to minimize errors.",
                "This technology powers self-driving cars and medical diagnoses.",
                "Mathematical architecture transforms raw data into intelligent action."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Color line 1
        self.lecture[0].set_color(YELLOW)

        # Neural Network Layers
        input_layer_nodes = VGroup(*[Circle(radius=0.15, color=BLUE, fill_opacity=0.5) for _ in range(3)])
        hidden_layer_nodes = VGroup(*[Circle(radius=0.15, color=GREEN, fill_opacity=0.5) for _ in range(4)])
        output_layer_nodes = VGroup(*[Circle(radius=0.15, color=RED, fill_opacity=0.5) for _ in range(2)])

        # Position nodes using grid (Issue 45: Shift to Col 2, Issue 46: Shift to Col 4)
        # Input layer: column 2
        self.place_at_grid(input_layer_nodes[0], 'B2')
        self.place_at_grid(input_layer_nodes[1], 'C2')
        self.place_at_grid(input_layer_nodes[2], 'D2')
        
        # Hidden layer: column 4
        self.place_at_grid(hidden_layer_nodes[0], 'B4')
        self.place_at_grid(hidden_layer_nodes[1], 'C4')
        self.place_at_grid(hidden_layer_nodes[2], 'D4')
        self.place_at_grid(hidden_layer_nodes[3], 'E4')

        # Output layer: column 5
        self.place_at_grid(output_layer_nodes[0], 'C5')
        self.place_at_grid(output_layer_nodes[1], 'D5')

        # Connections (Weights)
        weights_ih = VGroup()
        for i_node in input_layer_nodes:
            for h_node in hidden_layer_nodes:
                weights_ih.add(Line(i_node.get_center(), h_node.get_center(), stroke_width=1.5, color=GRAY))
        
        weights_ho = VGroup()
        for h_node in hidden_layer_nodes:
            for o_node in output_layer_nodes:
                weights_ho.add(Line(h_node.get_center(), o_node.get_center(), stroke_width=1.5, color=GRAY))

        self.play(
            Create(input_layer_nodes),
            Create(hidden_layer_nodes),
            Create(output_layer_nodes),
            Create(weights_ih),
            Create(weights_ho),
            run_time=2
        )

        # Adjust weights (learning animation)
        self.play(
            weights_ih.animate.set_color("#FF4500"),
            weights_ho.animate.set_color("#FF4500"),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Color line 2, reset line 1
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # Labels for output layer at column 6 (Issue 31: Integrate asset)
        # Pedestrian icon and text
        pedestrian_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pedestrian.svg", height=0.3, color=WHITE)
        pedestrian_text = Text("Pedestrian", font_size=16, color=WHITE)
        pedestrian_label = VGroup(pedestrian_icon, pedestrian_text).arrange(RIGHT, buff=0.1)
        
        mailbox_label = Text("Mailbox", font_size=16, color=WHITE)

        self.place_at_grid(pedestrian_label, 'C6')
        self.place_at_grid(mailbox_label, 'D6')

        # Identification animation
        self.play(
            FadeIn(pedestrian_label),
            FadeIn(mailbox_label)
        )
        
        # Highlighting logic: focus on 'Pedestrian' then 'Mailbox'
        self.play(
            output_layer_nodes[0].animate.scale(1.2).set_color(YELLOW),
            pedestrian_label.animate.scale(1.2).set_color(YELLOW),
            run_time=1
        )
        self.wait(1)
        
        self.play(
            output_layer_nodes[0].animate.scale(1/1.2).set_color(RED),
            pedestrian_label.animate.scale(1/1.2).set_color(WHITE),
            output_layer_nodes[1].animate.scale(1.2).set_color(YELLOW),
            mailbox_label.animate.scale(1.2).set_color(YELLOW),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Color line 3, reset line 2
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Reset output layer highlighting
        self.play(
            output_layer_nodes[1].animate.scale(1/1.2).set_color(RED),
            mailbox_label.animate.scale(1/1.2).set_color(WHITE),
        )

        # Summary text at the bottom area (Row F) (Issue 47: Increase scale)
        summary_text = Text("Mathematical Precision", font_size=28, slant=ITALIC, color=BLUE_B)
        self.place_in_area(summary_text, 'F1', 'F6', scale_factor=1.2)

        self.play(
            FadeIn(summary_text, shift=UP),
            run_time=2
        )
        self.wait(3)
