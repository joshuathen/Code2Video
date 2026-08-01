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
        self.setup_layout("The Biological Inspiration", [
            "Neural networks are inspired by the human brain.",
            "Interconnected neurons process information to make decisions.",
            "Computers mimic this structure to recognize complex patterns."
        ])

        # === Animation for Lecture Line 1 ===
        # Show a biological neuron schematic (#ADD8E6) morphing into a digital node (#90EE90)
        self.lecture[0].set_color("#ADD8E6")
        
        bio_color = "#ADD8E6"
        neuron_core = Circle(radius=0.4, color=bio_color, fill_opacity=0.3)
        self.place_at_grid(neuron_core, "C3")
        
        dendrites = VGroup(*[
            Line(neuron_core.get_center(), neuron_core.get_center() + 0.6 * np.array([np.cos(a), np.sin(a), 0]), color=bio_color)
            for a in np.linspace(0, 2*PI, 6, endpoint=False)
        ])
        axon = Line(neuron_core.get_center(), neuron_core.get_center() + np.array([0.8, 0, 0]), color=bio_color)
        biological_neuron = VGroup(neuron_core, dendrites, axon)
        
        self.play(FadeIn(biological_neuron))
        self.wait(1)
        
        digital_node_color = "#90EE90"
        digital_node = Circle(radius=0.4, color=digital_node_color, fill_opacity=0.5)
        self.place_at_grid(digital_node, "C3")
        
        self.play(
            ReplacementTransform(biological_neuron, digital_node),
            self.lecture[0].animate.set_color(digital_node_color)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Interconnected neurons process information. 
        # Animate signals pulsing through interconnected nodes.
        self.play(
            self.lecture[0].animate.set_color(GRAY),
            self.lecture[1].animate.set_color(WHITE)
        )
        
        # Input representation (following the outline example)
        kitten_input = RoundedRectangle(corner_radius=0.1, height=1.0, width=1.2, color=BLUE)
        kitten_text = Text("Kitten", font_size=16, color=BLUE)
        kitten_group = VGroup(kitten_input, kitten_text)
        self.place_at_grid(kitten_group, "B1")
        
        # Hidden Layer
        hidden_nodes = VGroup(*[
            Circle(radius=0.3, color=digital_node_color, fill_opacity=0.3) for _ in range(3)
        ])
        for i, node in enumerate(hidden_nodes):
            self.place_at_grid(node, f"{chr(ord('B')+i)}3")
            
        self.play(
            FadeIn(kitten_group),
            ReplacementTransform(digital_node, hidden_nodes[1]),
            FadeIn(hidden_nodes[0]),
            FadeIn(hidden_nodes[2])
        )
        
        # Connections from input to hidden
        conn_group = VGroup()
        for node in hidden_nodes:
            conn = Line(kitten_group.get_right(), node.get_left(), color=WHITE, stroke_width=2)
            conn_group.add(conn)
            
        self.play(Create(conn_group))
        
        # Labels 'Whiskers' and 'Sharpness' (Applying Fixes for Issue 32 and 33)
        label_whiskers = Text("Whiskers", font_size=14, color=YELLOW)
        label_sharpness = Text("Sharpness", font_size=14, color=YELLOW)
        
        # Fix 32: B2 -> A2, scale 0.8 to avoid overlap with connection lines
        self.place_at_grid(label_whiskers, "A2", scale_factor=0.8)
        # Fix 33: C2 -> C1, scale 0.8 to avoid overlap with connection lines
        self.place_at_grid(label_sharpness, "C1", scale_factor=0.8)
        
        self.play(Write(label_whiskers), Write(label_sharpness))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Computers mimic this structure to recognize complex patterns.
        # Output node glows and displays 'Pattern Recognized'.
        self.play(
            self.lecture[1].animate.set_color(GRAY),
            self.lecture[2].animate.set_color(WHITE)
        )
        
        output_color = "#FFD700"
        output_node = Circle(radius=0.4, color=WHITE, fill_opacity=0.2)
        self.place_at_grid(output_node, "C5")
        
        out_conns = VGroup()
        for node in hidden_nodes:
            conn = Line(node.get_right(), output_node.get_left(), color=WHITE, stroke_width=2)
            out_conns.add(conn)
            
        self.play(Create(out_conns), FadeIn(output_node))
        
        # Pulsing signals
        def pulse_along_lines(conns):
            dots = VGroup()
            for conn in conns:
                dot = Dot(conn.get_start(), radius=0.08, color=WHITE)
                dots.add(dot)
            self.play(*(dot.animate.move_to(conn.get_end()) for dot, conn in zip(dots, conns)), rate_func=linear, run_time=1)
            self.remove(dots)

        pulse_along_lines(conn_group)
        pulse_along_lines(out_conns)
        
        # Output glow and label (Applying Fix for Issue 34 and Storyboard text)
        pattern_label = Text("Pattern Recognized", font_size=16, color=output_color)
        # Fix 34: D5, scale 0.8 to prevent cramped appearance
        self.place_at_grid(pattern_label, "D5", scale_factor=0.8)
        
        self.play(
            output_node.animate.set_color(output_color).set_fill(output_color, opacity=0.8),
            Write(pattern_label),
            self.lecture[2].animate.set_color(output_color)
        )
        self.wait(2)
