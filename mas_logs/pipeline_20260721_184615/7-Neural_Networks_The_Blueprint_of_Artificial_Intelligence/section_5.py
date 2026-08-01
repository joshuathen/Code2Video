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

class Section5Scene(TeachingScene):
    def construct(self):
        # Setup Title and Lecture Lines
        title = "Information Flow: Forward Propagation"
        lines = [
            "Data flows forward through the network's layers.",
            "Each layer's output becomes the input for the next.",
            "Thousands of simple calculations cascade toward a final result.",
            "The output layer represents probabilities for different outcomes.",
            "High values indicate the network's most likely prediction."
        ]
        self.setup_layout(title, lines)

        # --- PREPARATION OF VISUAL ASSETS ---
        
        # Input Layer (3 nodes) - Shifted to Col 3 (Issue 35, 43)
        input_nodes = VGroup(*[Circle(radius=0.15, color="#00FFFF", stroke_width=2).set_fill(color="#00FFFF", opacity=0.3) for _ in range(3)]).arrange(DOWN, buff=0.5)
        self.place_in_area(input_nodes, "B3", "E3")
        input_label = Text("Input", font_size=20, color="#00FFFF")
        self.place_at_grid(input_label, "A3", scale_factor=0.8)

        # Hidden Layer (4 nodes) - Stays at Col 4
        hidden_nodes = VGroup(*[Circle(radius=0.15, color="#FFFFFF", stroke_width=2).set_fill(color="#FFFFFF", opacity=0.3) for _ in range(4)]).arrange(DOWN, buff=0.4)
        self.place_in_area(hidden_nodes, "B4", "E4")
        hidden_label = Text("Hidden", font_size=20, color="#FFFFFF")
        self.place_at_grid(hidden_label, "A4", scale_factor=0.8)

        # Output Layer (10 nodes) - Shifted to Col 5 (Issue 36, 43)
        output_nodes = VGroup(*[Circle(radius=0.1, color="#FFFFFF", stroke_width=1.5).set_fill(color="#FFFFFF", opacity=0.3) for _ in range(10)]).arrange(DOWN, buff=0.1)
        self.place_in_area(output_nodes, "B5", "F5")
        output_label = Text("Output", font_size=20, color="#FFC0CB")
        self.place_at_grid(output_label, "A5", scale_factor=0.8)

        # Static Connections (Background)
        conn1 = VGroup()
        for i_n in input_nodes:
            for h_n in hidden_nodes:
                conn1.add(Line(i_n.get_right(), h_n.get_left(), stroke_width=1, color="#FFFFFF", stroke_opacity=0.1))

        conn2 = VGroup()
        for h_n in hidden_nodes:
            for o_n in output_nodes:
                conn2.add(Line(h_n.get_right(), o_n.get_left(), stroke_width=1, color="#FFFFFF", stroke_opacity=0.1))

        self.add(input_nodes, hidden_nodes, output_nodes, input_label, hidden_label, output_label, conn1, conn2)

        # === Animation for Lecture Line 1 ===
        # Line: "Data flows forward through the network's layers."
        # Use Asset for '7' (Issue 21, 43) and place at C2 (Issue 34, 43)
        input_digit = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/seven.svg", color="#00FFFF")
        self.place_at_grid(input_digit, "C2", scale_factor=1.0)
        
        self.play(
            self.lecture[0].animate.set_color("#00FFFF"),
            FadeIn(input_digit)
        )
        self.play(
            input_digit.animate.move_to(input_nodes.get_center()).scale(0.5).set_fill(opacity=0),
            *[n.animate.set_fill(color="#00FFFF", opacity=1) for n in input_nodes],
            run_time=1.5
        )
        self.play(Flash(input_nodes, color="#00FFFF"))
        self.wait(2.0)

        # === Animation for Lecture Line 2 ===
        # Line: "Each layer's output becomes the input for the next."
        # Animated white glowing arrows from input to hidden
        glowing_conn1 = VGroup()
        for i_n in input_nodes:
            for h_n in hidden_nodes:
                glowing_conn1.add(Line(i_n.get_right(), h_n.get_left(), stroke_width=2, color="#FFFFFF"))
        
        self.play(
            self.lecture[1].animate.set_color("#FFFFFF"),
            Create(glowing_conn1),
            *[n.animate.set_fill(color="#FFFFFF", opacity=1) for n in hidden_nodes],
            run_time=2
        )
        self.wait(2.0)

        # === Animation for Lecture Line 3 ===
        # Line: "Thousands of simple calculations cascade toward a final result."
        # Animated white glowing arrows from hidden to output
        glowing_conn2 = VGroup()
        for h_n in hidden_nodes:
            for o_n in output_nodes:
                glowing_conn2.add(Line(h_n.get_right(), o_n.get_left(), stroke_width=2, color="#FFFFFF"))

        self.play(
            self.lecture[2].animate.set_color("#FFFFFF"),
            Create(glowing_conn2),
            *[n.animate.set_fill(color="#FFFFFF", opacity=1) for n in output_nodes],
            run_time=2
        )
        self.wait(2.0)

        # === Animation for Lecture Line 4 ===
        # Line: "The output layer represents probabilities for different outcomes."
        self.play(
            self.lecture[3].animate.set_color("#FFC0CB"),
            output_label.animate.scale(1.1).set_color("#FFC0CB"),
            Indicate(output_nodes, color="#FFC0CB")
        )
        self.wait(2.0)

        # === Animation for Lecture Line 5 ===
        # Line: "High values indicate the network's most likely prediction."
        # Node index 7 corresponds to digit "7".
        # Since there are 10 nodes (0-9), index 7 is the 8th node.
        target_node = output_nodes[7]
        target_prob = Text("0.98", font_size=16, color="#00FF00")
        # Place probability text in Col 6
        target_prob.next_to(target_node, RIGHT, buff=0.1)
        
        self.play(
            self.lecture[4].animate.set_color("#00FF00"),
            target_node.animate.set_color("#00FF00").scale(1.4).set_fill(opacity=1),
            FadeIn(target_prob)
        )
        self.play(Indicate(target_node, color="#00FF00"), run_time=1.5)
        self.wait(1.5)
