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
        # Setup the layout
        self.setup_layout(
            "Forward Propagation: The Complete Flow",
            [
                "Data enters the network at the initial input layer.",
                "Signals travel through connections, weighted by their learned importance.",
                "Each neuron sums its inputs and applies an activation.",
                "This process repeats until reaching the final output layer.",
                "The network produces a probability for its final prediction."
            ]
        )

        # Colors for consistency
        color_input = BLUE_A
        color_signals = GREEN_A
        color_math = YELLOW_A
        color_repeat = ORANGE
        color_prediction = "#2ECC71" # Specific green from VideoCritic

        # Assets
        ball_asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg"
        toy_asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/toy.svg"

        # === Animation for Lecture Line 1 ===
        # Data enters the network at the initial input layer.
        self.play(self.lecture[0].animate.set_color(color_input))
        
        # Build Network Architecture
        # Input Layer (Col 2) - Issue 36: Move to Col 2 to avoid crowding
        input_nodes = VGroup(*[Circle(radius=0.15, color=WHITE, fill_opacity=0.2) for _ in range(3)])
        self.place_at_grid(input_nodes[0], 'B2')
        self.place_at_grid(input_nodes[1], 'C2')
        self.place_at_grid(input_nodes[2], 'D2')
        
        # Hidden Layer (Col 4) - Issue 38: Move to Col 4 for expansion
        hidden_nodes = VGroup(*[Circle(radius=0.15, color=WHITE, fill_opacity=0.2) for _ in range(4)])
        self.place_at_grid(hidden_nodes[0], 'B4')
        self.place_at_grid(hidden_nodes[1], 'C4')
        self.place_at_grid(hidden_nodes[2], 'D4')
        self.place_at_grid(hidden_nodes[3], 'E4')
        
        # Output Layer (Col 6) - Issue 38: Move to Col 6 for expansion
        output_nodes = VGroup(*[Circle(radius=0.15, color=WHITE, fill_opacity=0.2) for _ in range(2)])
        self.place_at_grid(output_nodes[0], 'C6')
        self.place_at_grid(output_nodes[1], 'D6')
        
        # Weights (Gray lines)
        weights_in_hid = VGroup()
        for i_n in input_nodes:
            for h_n in hidden_nodes:
                weights_in_hid.add(Line(i_n.get_right(), h_n.get_left(), stroke_width=1, color=GRAY_C))
        
        weights_hid_out = VGroup()
        for h_n in hidden_nodes:
            for o_n in output_nodes:
                weights_hid_out.add(Line(h_n.get_right(), o_n.get_left(), stroke_width=1, color=GRAY_C))
        
        self.add(weights_in_hid, weights_hid_out, input_nodes, hidden_nodes, output_nodes)
        
        # Ball "data" entry [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg] - Issue 21
        ball = SVGMobject(ball_asset_path).scale(0.2).set_color(color_input)
        ball.move_to(self.grid['C2'] + LEFT * 1.5)
        self.play(FadeIn(ball))
        self.play(ball.animate.move_to(self.grid['C2']), input_nodes[1].animate.set_color(color_input))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Signals travel through connections, weighted by their learned importance.
        self.play(self.lecture[1].animate.set_color(color_signals))
        
        # Highlights on connections (active path)
        signal_paths = VGroup(
            Line(input_nodes[1].get_right(), hidden_nodes[1].get_left(), color=color_signals, stroke_width=4),
            Line(input_nodes[1].get_right(), hidden_nodes[2].get_left(), color=color_signals, stroke_width=4)
        )
        self.play(Create(signal_paths))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Each neuron sums its inputs and applies an activation.
        self.play(self.lecture[2].animate.set_color(color_math))
        
        # Issue 37: Move formula to A4, scale 0.7 to avoid overlap
        formula = MathTex(r"z = \sum w_i x_i + b", font_size=24, color=color_math)
        self.place_at_grid(formula, 'A4', scale_factor=0.7)
        
        self.play(
            hidden_nodes[1].animate.set_color(color_math), 
            hidden_nodes[2].animate.set_color(color_math),
            Write(formula)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # This process repeats until reaching the final output layer.
        self.play(self.lecture[3].animate.set_color(color_repeat))
        
        repeat_paths = VGroup(
            Line(hidden_nodes[1].get_right(), output_nodes[0].get_left(), color=color_repeat, stroke_width=4),
            Line(hidden_nodes[2].get_right(), output_nodes[0].get_left(), color=color_repeat, stroke_width=4)
        )
        self.play(Create(repeat_paths), output_nodes[0].animate.set_color(color_repeat))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The network produces a probability for its final prediction.
        self.play(self.lecture[4].animate.set_color(color_prediction))
        
        # Issue 38 & 21: Prediction Label with Asset at B6, scale 0.8
        toy_icon = SVGMobject(toy_asset_path).scale(0.3).set_color(color_prediction)
        toy_text = Text("Toy: 98%", font_size=20, color=color_prediction)
        prediction_label = VGroup(toy_icon, toy_text).arrange(RIGHT, buff=0.2)
        self.place_at_grid(prediction_label, 'B6', scale_factor=0.8)
        
        self.play(FadeIn(prediction_label))
        self.play(Indicate(prediction_label))
        self.wait(2)
