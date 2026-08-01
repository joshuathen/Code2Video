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
        title_str = "The Fundamental Theorem: The Great Bridge"
        lecture_lines = [
            "Differentiation and integration are actually perfect opposites.",
            "One finds the slope, the other finds the area.",
            "A climber’s steepness at any point determines their height.",
            "Tracking growth rate reveals the total growth achieved.",
            "This bridge connects the rate of change to accumulation."
        ]
        self.setup_layout(title_str, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Split screen: Left side shows a derivative graph, right shows an integral graph.
        self.lecture[0].set_color("#FFFF00")
        
        # Derivative Graph: f(x) = 2x
        deriv_axes = Axes(x_range=[0, 3], y_range=[0, 6], axis_config={"include_tip": False}, x_length=2.5, y_length=2.5)
        deriv_graph = deriv_axes.plot(lambda x: 2*x, color=YELLOW)
        deriv_label = Text("Derivative", font_size=16, color=YELLOW)
        deriv_group = VGroup(deriv_axes, deriv_graph, deriv_label).arrange(DOWN, buff=0.2)
        self.place_in_area(deriv_group, "A1", "C3", scale_factor=0.8)
        
        # Integral Graph: F(x) = x^2
        int_axes = Axes(x_range=[0, 3], y_range=[0, 9], axis_config={"include_tip": False}, x_length=2.5, y_length=2.5)
        int_graph = int_axes.plot(lambda x: x**2, color=TEAL)
        int_label = Text("Integral", font_size=16, color=TEAL)
        int_group = VGroup(int_axes, int_graph, int_label).arrange(DOWN, buff=0.2)
        self.place_in_area(int_group, "A4", "F3", scale_factor=0.8)

        self.play(Create(deriv_group), Create(int_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight the slope on the left (#FFFF00) and the area on the right (#00FFFF).
        self.lecture[1].set_color("#FFFF00")
        
        # Slope highlight
        p1 = deriv_axes.c2p(1, 2)
        p2 = deriv_axes.c2p(2, 4)
        slope_line = Line(p1, p2, color=YELLOW, stroke_width=6)
        slope_label = Text("Slope", font_size=14, color=YELLOW)
        self.place_at_grid(slope_label, "B3", scale_factor=0.8)

        # Area highlight
        area = int_axes.get_area(int_graph, x_range=[0, 2], color=TEAL, opacity=0.3)
        area_label = Text("Area", font_size=14, color=TEAL)
        self.place_at_grid(area_label, "B6", scale_factor=0.8)

        self.play(Create(slope_line), Write(slope_label), FadeIn(area), Write(area_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show a climber icon (#FF8C00) on a mountain; display steepness arrows.
        self.lecture[2].set_color("#FFFF00")
        
        mountain_axes = Axes(x_range=[0, 4], y_range=[0, 4], x_length=3, y_length=2)
        mountain_curve = mountain_axes.plot(lambda x: np.sin(x) + 1, color=GREY)
        climber = Triangle(color="#FF8C00", fill_opacity=1).scale(0.1)
        
        # Steepness arrows
        arrow1 = Arrow(start=ORIGIN, end=UP*0.5, color="#FF8C00").scale(0.5)
        arrow2 = Arrow(start=ORIGIN, end=RIGHT*0.5+UP*0.3, color="#FF8C00").scale(0.5)
        steepness_group = VGroup(arrow1, arrow2).arrange(RIGHT, buff=0.5)
        
        mountain_viz = VGroup(mountain_axes, mountain_curve)
        self.place_in_area(mountain_viz, "D1", "F3", scale_factor=0.8) 
        self.place_at_grid(climber, "E1", scale_factor=1.0)
        self.place_at_grid(steepness_group, "F1", scale_factor=1.0)

        # Movement animation
        tracker = ValueTracker(0)
        climber.add_updater(lambda m: m.move_to(mountain_axes.c2p(tracker.get_value(), np.sin(tracker.get_value()) + 1) + UP*0.2))

        self.play(Create(mountain_viz), FadeIn(climber), Create(steepness_group))
        self.play(tracker.animate.set_value(3), run_time=2)
        climber.clear_updaters()
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Show a growing tree icon (#00FF00) whose height matches the integral area.
        self.lecture[3].set_color("#FFFF00")
        
        tree_trunk = Rectangle(width=0.2, height=0.5, color="#8B4513", fill_opacity=1)
        tree_foliage = VGroup(
            Triangle(color="#00FF00", fill_opacity=1).scale(0.4),
            Triangle(color="#00FF00", fill_opacity=1).scale(0.4).shift(UP*0.2),
            Triangle(color="#00FF00", fill_opacity=1).scale(0.4).shift(UP*0.4)
        )
        tree = VGroup(tree_trunk, tree_foliage).arrange(UP, buff=0)
        self.place_in_area(tree, "D4", "F6", scale_factor=1.0)
        
        growth_tracker = ValueTracker(0.1)
        tree.add_updater(lambda m: m.set_height(growth_tracker.get_value(), stretch=True))

        growth_label = Text("Accumulated Growth", font_size=16, color="#00FF00")
        self.place_at_grid(growth_label, "D6", scale_factor=0.8)

        self.play(FadeIn(tree), Write(growth_label))
        self.play(growth_tracker.animate.set_value(2), run_time=2)
        tree.clear_updaters()
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Draw a bridge (#FFFFFF) connecting the concepts of 'Slope' and 'Area'.
        self.lecture[4].set_color("#FFFF00")
        
        bridge_base = Line(self.grid["B3"], self.grid["B6"], color=WHITE, stroke_width=4)
        bridge_label = Text("THE BRIDGE", font_size=20, color=WHITE)
        self.place_in_area(bridge_label, "B3", "B6", scale_factor=1.0)
        bridge_label.shift(UP*0.5)

        self.play(Create(bridge_base), Write(bridge_label))
        self.wait(2)