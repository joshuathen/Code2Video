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
        lecture_lines = [
            "Meet the artificial neuron, the digital version of a cell.",
            "It receives multiple inputs, each carrying specific data values.",
            "Every input has a weight representing its relative importance.",
            "Higher weights mean the input strongly influences the result.",
            "These weights are visually represented by varying line thicknesses."
        ]
        self.setup_layout("The Artificial Neuron: Input and Weight", lecture_lines)
        
        # Colors
        NODE_COLOR = "#D3D3D3"
        HIGHLIGHT_COLOR = "#FFFF00"
        ARROW_COLOR = "#A9A9A9"
        TEXT_COLOR = WHITE

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT_COLOR))
        neuron_node = Circle(radius=0.5, color=NODE_COLOR, fill_opacity=0.2)
        neuron_label = Text("Neuron", font_size=24, color=TEXT_COLOR)
        
        # Issue 25: Move neuron_node to C6
        self.place_at_grid(neuron_node, "C6")
        # Issue 26: Move neuron_label to D6, scale 0.8
        self.place_at_grid(neuron_label, "D6", scale_factor=0.8)
        
        self.play(Create(neuron_node), Write(neuron_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        input_nodes = VGroup()
        input_labels = VGroup()
        grid_positions = ["B2", "C2", "D2"]
        for i, pos in enumerate(grid_positions):
            node = Circle(radius=0.3, color=NODE_COLOR, fill_opacity=0.1)
            label = Text(f"x{i+1}", font_size=24, color=TEXT_COLOR)
            self.place_at_grid(node, pos)
            label.next_to(node, LEFT, buff=0.2)
            input_nodes.add(node)
            input_labels.add(label)
            
        self.play(Create(input_nodes), Write(input_labels))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        thicknesses = [8, 2, 5]
        arrows = VGroup()
        
        for i, node in enumerate(input_nodes):
            arrow = Line(
                node.get_right(), 
                neuron_node.get_left(), 
                color=ARROW_COLOR, 
                stroke_width=thicknesses[i]
            ).add_tip(tip_length=0.15)
            arrows.add(arrow)
            
        # Issue 27: Position weight_labels in area B5 to D5 with scale 0.7
        weight_labels = VGroup(*[Text(f"w{i+1}", font_size=24, color=TEXT_COLOR) for i in range(3)])
        weight_labels.arrange(DOWN, buff=0.7)
        self.place_in_area(weight_labels, "B5", "D5", scale_factor=0.7)
            
        self.play(Create(arrows), Write(weight_labels))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        # Highlight w1 (the thickest one)
        self.play(
            arrows[0].animate.set_color(HIGHLIGHT_COLOR),
            weight_labels[0].animate.set_color(HIGHLIGHT_COLOR),
            Indicate(arrows[0], scale_factor=1.1)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        # Emphasize thickness difference by briefly highlighting other arrows
        self.play(
            arrows[1].animate.set_color(HIGHLIGHT_COLOR),
            arrows[2].animate.set_color(HIGHLIGHT_COLOR),
            run_time=0.5
        )
        self.play(
            arrows[1].animate.set_color(ARROW_COLOR),
            arrows[2].animate.set_color(ARROW_COLOR),
            run_time=0.5
        )
        
        self.wait(2)
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(1)
