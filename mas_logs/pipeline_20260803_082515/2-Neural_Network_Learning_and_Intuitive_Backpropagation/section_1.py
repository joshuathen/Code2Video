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
        title = "The Big Picture: Learning as Fine-Tuning"
        lines = [
            "Neural networks learn by adjusting internal parameters called weights.",
            "Think of weights as dials on a machine.",
            "Tuning these dials helps the network produce correct outputs."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # Neural networks learn by adjusting internal parameters called weights.
        self.play(self.lecture[0].animate.set_color(BLUE))
        
        # Nodes
        input_node1 = Circle(radius=0.3, color=WHITE)
        input_node2 = Circle(radius=0.3, color=WHITE)
        output_node = Circle(radius=0.3, color=WHITE)
        
        # Grid Fix (Issue 24): B2/D2 -> B3/D3
        self.place_at_grid(input_node1, "B3")
        self.place_at_grid(input_node2, "D3")
        self.place_at_grid(output_node, "C5")
        
        # Connections
        line1 = Line(input_node1.get_right(), output_node.get_left(), color=WHITE)
        line2 = Line(input_node2.get_right(), output_node.get_left(), color=WHITE)
        
        # Weights Label
        weights_label = Text("Weights", font_size=24, color=WHITE)
        # Grid Fix (Issue 25): Area B3-D4 -> Grid C3, scale 0.7
        self.place_at_grid(weights_label, "C3", scale_factor=0.7)
        
        self.play(Create(input_node1), Create(input_node2), Create(output_node))
        self.play(Create(line1), Create(line2))
        self.play(Write(weights_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Think of weights as dials on a machine.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(GREEN)
        )
        
        # Flour and Sugar icons entering
        flour_label = Text("Flour", font_size=16, color="#FFD700")
        sugar_label = Text("Sugar", font_size=16, color="#FFD700")
        
        # Using simple shapes for icons
        flour_icon = VGroup(Square(side_length=0.4, color="#FFD700"), flour_label).arrange(DOWN, buff=0.1)
        sugar_icon = VGroup(Square(side_length=0.4, color="#FFD700"), sugar_label).arrange(DOWN, buff=0.1)
        
        # Grid Fix (Issue 23): B1/D1 -> B2/D2
        self.place_at_grid(flour_icon, "B2")
        self.place_at_grid(sugar_icon, "D2")
        
        # Dial icon
        dial_circle = Circle(radius=0.4, color="#00FF00")
        dial_pointer = Line(dial_circle.get_center(), dial_circle.get_top(), color="#00FF00")
        dial = VGroup(dial_circle, dial_pointer)
        self.place_at_grid(dial, "C4")
        
        self.play(
            FadeIn(flour_icon, shift=RIGHT),
            FadeIn(sugar_icon, shift=RIGHT)
        )
        self.play(Create(dial))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Tuning these dials helps the network produce correct outputs.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Burnt Cookie at output
        burnt_label = Text("Burnt", font_size=16, color="#8B4513")
        burnt_cookie = VGroup(Circle(radius=0.3, color="#8B4513", fill_opacity=0.8), burnt_label).arrange(DOWN, buff=0.1)
        self.place_at_grid(burnt_cookie, "C6")
        
        # Perfect Cookie
        perfect_label = Text("Perfect", font_size=16, color="#F5DEB3")
        perfect_cookie = VGroup(Circle(radius=0.3, color="#F5DEB3", fill_opacity=0.8), perfect_label).arrange(DOWN, buff=0.1)
        self.place_at_grid(perfect_cookie, "C6")
        
        self.play(FadeIn(burnt_cookie))
        
        # Rotate dial and scale weights
        self.play(
            Rotate(dial, angle=PI/2),
            weights_label.animate.scale(1.2),
            run_time=2
        )
        
        # Transform cookie
        self.play(Transform(burnt_cookie, perfect_cookie))
        self.wait(2)
