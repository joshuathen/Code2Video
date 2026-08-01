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
        lecture_lines = [
            "Convolution is essentially a weighted local average.",
            "It is a vital tool for signal processing and AI.",
            "This simple operation shapes our digital world."
        ]
        self.setup_layout("Summary and Conclusion", lecture_lines)
        
        # Colors for highlights
        HIGHLIGHT_COLOR = "#FFD700"  # Gold
        CYAN_COLOR = "#00FFFF"
        
        # === Animation for Lecture Line 1 ===
        # Convolution is essentially a weighted local average.
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT_COLOR))
        
        # Set up a group of points representing discrete values
        points = VGroup(*[Dot(radius=0.1, color=WHITE) for _ in range(7)]).arrange(RIGHT, buff=0.5)
        self.place_in_area(points, "D1", "D6")
        
        # Gaussian weight curve
        gaussian_curve = FunctionGraph(
            lambda x: 1.5 * np.exp(-x**2),
            x_range=[-2, 2],
            color=HIGHLIGHT_COLOR
        ).move_to(points.get_center() + UP * 0.5)
        
        # Arrow pointing to central weight
        center_point = points[3]
        highlight_ring = Circle(radius=0.2, color=HIGHLIGHT_COLOR).move_to(center_point.get_center())
        
        self.play(Create(points))
        self.play(Create(gaussian_curve))
        self.play(Create(highlight_ring))
        self.wait(1.5)
        
        # Fade out Line 1 elements
        self.play(
            FadeOut(points), FadeOut(gaussian_curve), FadeOut(highlight_ring),
            self.lecture[0].animate.set_color(WHITE)
        )

        # === Animation for Lecture Line 2 ===
        # It is a vital tool for signal processing and AI.
        self.play(self.lecture[1].animate.set_color(CYAN_COLOR))
        
        # Asset: Digital Signal (Oscilloscope)
        oscill_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/oscill.svg")
        oscill_icon.set_color(CYAN_COLOR)
        self.place_in_area(oscill_icon, "A1", "C6", scale_factor=1.2)
        
        # Representing AI (Neural Network nodes and edges)
        nn_nodes = VGroup()
        layers = [2, 3, 2]
        node_radius = 0.1
        for i, count in enumerate(layers):
            layer = VGroup(*[Dot(radius=node_radius, color=CYAN_COLOR) for _ in range(count)])
            layer.arrange(DOWN, buff=0.4)
            layer.shift(RIGHT * i * 0.8)
            nn_nodes.add(layer)
        
        nn_edges = VGroup()
        for i in range(len(layers)-1):
            for n1 in nn_nodes[i]:
                for n2 in nn_nodes[i+1]:
                    edge = Line(n1.get_center(), n2.get_center(), stroke_width=1, color=CYAN_COLOR, stroke_opacity=0.5)
                    nn_edges.add(edge)
        
        nn_group = VGroup(nn_edges, nn_nodes)
        self.place_in_area(nn_group, "D1", "F6", scale_factor=1.2)
        
        self.play(FadeIn(oscill_icon))
        self.play(Create(nn_group))
        self.wait(2)
        
        # Fade out Line 2 elements
        self.play(
            FadeOut(oscill_icon), FadeOut(nn_group),
            self.lecture[1].animate.set_color(WHITE)
        )

        # === Animation for Lecture Line 3 ===
        # This simple operation shapes our digital world.
        self.play(self.lecture[2].animate.set_color(WHITE)) # Keeping it neutral or white as requested
        
        # Split Screen Comparison
        v_line = Line(self.grid["A4"] + UP*0.5, self.grid["F4"] + DOWN*0.5, color=WHITE)
        
        # Left side: Blurred Fox
        fox_base = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/fox.svg")
        # Simulating blur with multiple layered copies
        blurred_fox = VGroup(*[
            fox_base.copy().set_opacity(0.15).shift(np.array([x, y, 0]))
            for x in np.linspace(-0.08, 0.08, 3)
            for y in np.linspace(-0.08, 0.08, 3)
        ])
        self.place_in_area(blurred_fox, "A1", "F3", scale_factor=1.0)
        
        # Right side: Sharp Fox
        sharp_fox = fox_base.copy()
        self.place_in_area(sharp_fox, "A4", "F6", scale_factor=1.0)
        
        # Labels
        blur_label = Text("Blurred", font_size=18, color=WHITE)
        sharp_label = Text("Sharpened", font_size=18, color=WHITE)
        self.place_at_grid(blur_label, "F2", scale_factor=1.0)
        self.place_at_grid(sharp_label, "F5", scale_factor=1.0)
        
        self.play(Create(v_line))
        self.play(FadeIn(blurred_fox), Write(blur_label))
        self.wait(0.5)
        self.play(FadeIn(sharp_fox), Write(sharp_label))
        
        self.wait(3)
