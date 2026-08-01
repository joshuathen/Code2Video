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
        # Define content
        title = "Layered Architecture: Input, Hidden, and Output"
        lines = [
            "Neurons are organized into a tiered assembly line.",
            "The input layer receives the raw digital data.",
            "Hidden layers extract increasingly abstract features.",
            "Each layer builds upon the findings of the previous.",
            "Finally, the output layer delivers the network's prediction."
        ]
        
        # Setup layout
        self.setup_layout(title, lines)

        # Colors
        COLOR_INPUT = "#ADD8E6"
        COLOR_HIDDEN = "#98FB98"
        COLOR_OUTPUT = "#FFA07A"
        COLOR_GOLD = "#FFD700"
        COLOR_CONNECTIONS = GRAY_C

        # === Animation for Lecture Line 1 ===
        # Show three distinct layers of nodes
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        input_layer = VGroup(*[Circle(radius=0.2, color=COLOR_INPUT, fill_opacity=0.3, stroke_width=2) for _ in range(4)])
        hidden_layer = VGroup(*[Circle(radius=0.2, color=COLOR_HIDDEN, fill_opacity=0.3, stroke_width=2) for _ in range(4)])
        output_layer = VGroup(*[Circle(radius=0.2, color=COLOR_OUTPUT, fill_opacity=0.3, stroke_width=2) for _ in range(2)])

        input_positions = ["B2", "C2", "D2", "E2"]
        hidden_positions = ["B4", "C4", "D4", "E4"]
        output_positions = ["C5", "D5"]

        for node, pos in zip(input_layer, input_positions):
            self.place_at_grid(node, pos)
        for node, pos in zip(hidden_layer, hidden_positions):
            self.place_at_grid(node, pos)
        for node, pos in zip(output_layer, output_positions):
            self.place_at_grid(node, pos)

        # Create connections (edges)
        connections1 = VGroup()
        for in_node in input_layer:
            for hid_node in hidden_layer:
                line = Line(in_node.get_center(), hid_node.get_center(), stroke_width=1, color=COLOR_CONNECTIONS, stroke_opacity=0.2)
                connections1.add(line)

        connections2 = VGroup()
        for hid_node in hidden_layer:
            for out_node in output_layer:
                line = Line(hid_node.get_center(), out_node.get_center(), stroke_width=1, color=COLOR_CONNECTIONS, stroke_opacity=0.2)
                connections2.add(line)
        
        self.play(
            FadeIn(input_layer, shift=RIGHT),
            FadeIn(hidden_layer, shift=RIGHT),
            FadeIn(output_layer, shift=RIGHT),
            Create(connections1),
            Create(connections2),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The input layer receives the raw digital data.
        self.play(self.lecture[1].animate.set_color(COLOR_INPUT))
        self.play(
            LaggedStart(*[node.animate.set_fill(opacity=0.8).scale(1.2) for node in input_layer], lag_ratio=0.1),
            run_time=1
        )
        self.play(
            *[node.animate.set_fill(opacity=0.3).scale(1/1.2) for node in input_layer],
            run_time=0.5
        )

        # === Animation for Lecture Line 3 ===
        # Hidden layers extract increasingly abstract features.
        self.play(self.lecture[2].animate.set_color(COLOR_HIDDEN))
        self.play(
            LaggedStart(*[node.animate.set_fill(opacity=0.8).scale(1.2) for node in hidden_layer], lag_ratio=0.1),
            run_time=1
        )
        self.play(
            *[node.animate.set_fill(opacity=0.3).scale(1/1.2) for node in hidden_layer],
            run_time=0.5
        )

        # === Animation for Lecture Line 4 ===
        # Each layer builds upon the findings of the previous.
        self.play(self.lecture[3].animate.set_color(WHITE))
        
        # Glow propagation using ShowPassingFlash
        glow_lines = VGroup(*connections1, *connections2)
        self.play(
            ShowPassingFlash(
                glow_lines.copy().set_color(WHITE).set_stroke(width=3, opacity=1),
                time_width=0.5,
                run_time=2
            )
        )

        # === Animation for Lecture Line 5 ===
        # Finally, the output layer delivers the network's prediction.
        self.play(self.lecture[4].animate.set_color(COLOR_GOLD))
        
        target_node = output_layer[0] # Node at C5
        
        # Fix label position according to issue 41
        label_b = Text("B", font_size=24, color=COLOR_GOLD)
        self.place_at_grid(label_b, 'B5', scale_factor=0.7)
        
        self.play(
            target_node.animate.set_color(COLOR_GOLD).set_fill(COLOR_GOLD, opacity=1).scale(1.3),
            Write(label_b),
            run_time=1.5
        )
        self.wait(3)
