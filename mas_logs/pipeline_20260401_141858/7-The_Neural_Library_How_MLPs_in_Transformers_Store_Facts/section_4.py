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
        # Colors for matching lecture lines
        COLOR_1 = "#FFFF00" # Yellow
        COLOR_2 = "#00FF00" # Green
        COLOR_3 = "#FF6666" # Light Red

        self.setup_layout(
            "The Activation Trigger (The ReLU Gate)", 
            [
                "The activation function acts as a selective gate.", 
                "Only strong pattern matches trigger the factual retrieval.", 
                "Irrelevant information is filtered out by returning zero."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_1)
        
        # Create ReLU Graph
        axes = Axes(
            x_range=[-2.5, 2.5, 1],
            y_range=[-0.5, 2.5, 1],
            axis_config={"include_tip": True, "color": GREY},
            x_length=4,
            y_length=3
        )
        # Fix Issue 50: Reposition axes to avoid clutter
        self.place_in_area(axes, "B2", "E5", scale_factor=0.8)
        
        # Fix Issue 51: Reposition label to be centered horizontally relative to axes
        relu_label = Text("f(x) = max(0, x)", color=COLOR_1, font_size=24)
        self.place_at_grid(relu_label, "A4", scale_factor=1.0)
        
        relu_graph = axes.plot(
            lambda x: max(0, x),
            x_range=[-2.5, 2.5],
            color=COLOR_1
        )
        
        self.play(Create(axes), Write(relu_label))
        self.play(Create(relu_graph), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(COLOR_2)
        
        # Neuron representation
        neuron_circle = Circle(radius=0.5, color=GREY, fill_opacity=0.2)
        neuron_label = Text("Neuron", font_size=16, color=WHITE)
        neuron = VGroup(neuron_circle, neuron_label)
        self.place_at_grid(neuron, "D6", scale_factor=1.0)
        
        # Threshold line
        threshold_line = DashedLine(axes.c2p(0, 0), axes.c2p(0, 2), color=WHITE)
        threshold_label = Text("Threshold (0)", font_size=14, color=WHITE).next_to(threshold_line, DOWN, buff=0.1)
        
        # Animation: Input signal moving
        input_dot = Dot(color=COLOR_2)
        input_dot.move_to(axes.c2p(-2, 0))
        
        self.play(FadeIn(neuron), Create(threshold_line), Write(threshold_label))
        
        # Signal match triggers firing
        self.play(
            input_dot.animate.move_to(axes.c2p(1.5, 1.5)),
            run_time=2,
            rate_func=linear
        )
        
        # "Firing" flash
        flash = Flash(neuron_circle, color=COLOR_2, flash_radius=0.6, num_lines=12)
        self.play(
            neuron_circle.animate.set_fill(COLOR_2, opacity=0.8),
            flash,
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(COLOR_3)
        
        # Assets for Line 3 (Issue 39)
        tower_svg = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/tower.svg", color=COLOR_2)
        blueprint_svg = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/blueprint.svg", color=COLOR_3)
        
        self.place_at_grid(blueprint_svg, "F2", scale_factor=0.3)
        self.place_at_grid(tower_svg, "F6", scale_factor=0.3)
        
        # Labels for contrast (Issue 52)
        tower_text = Text("Eiffel Tower", font_size=18, color=COLOR_2)
        blueprint_text = Text("Eiffel Blueprint", font_size=18, color=COLOR_3)
        
        self.place_at_grid(blueprint_text, 'F3', scale_factor=0.9)
        self.place_at_grid(tower_text, 'F5', scale_factor=0.9)
        
        # Show contrast on graph
        blueprint_dot = Dot(axes.c2p(-1.5, 0), color=COLOR_3)
        tower_dot = Dot(axes.c2p(1.8, 1.8), color=COLOR_2)
        
        # Arrows pointing to results
        arrow_bp = Arrow(blueprint_text.get_top(), axes.c2p(-1.5, 0), color=COLOR_3, buff=0.1)
        arrow_tw = Arrow(tower_text.get_top(), axes.c2p(1.8, 1.8), color=COLOR_2, buff=0.1)
        
        # Result Labels
        zero_label = Text("Output: 0", font_size=16, color=COLOR_3).next_to(blueprint_dot, UP, buff=0.2)
        active_label = Text("Output: 1.8", font_size=16, color=COLOR_2).next_to(tower_dot, LEFT, buff=0.2)
        
        # Execution of contrast: Irrelevant input
        self.play(Write(blueprint_text), FadeIn(blueprint_svg), FadeIn(blueprint_dot), Create(arrow_bp))
        self.play(Write(zero_label))
        self.play(neuron_circle.animate.set_fill(COLOR_3, opacity=0.1))
        self.wait(1)

        # Execution of contrast: Relevant input
        self.play(Write(tower_text), FadeIn(tower_svg), FadeIn(tower_dot), Create(arrow_tw))
        self.play(Write(active_label))
        self.play(
            neuron_circle.animate.set_fill(COLOR_2, opacity=0.8),
            Flash(neuron_circle, color=COLOR_2)
        )
        self.wait(2)
