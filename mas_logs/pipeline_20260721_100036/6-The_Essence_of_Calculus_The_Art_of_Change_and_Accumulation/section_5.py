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
        
        # Derivative Graph: f(x) = 2 (Constant rate)
        deriv_axes = Axes(x_range=[0, 4], y_range=[0, 4], x_length=2.5, y_length=2.0, 
                          axis_config={"include_tip": False, "color": BLUE_E})
        deriv_graph = deriv_axes.plot(lambda x: 2, x_range=[0, 3], color=YELLOW)
        deriv_label = Text("Derivative (Rate)", font_size=18, color=YELLOW)
        deriv_vg = VGroup(deriv_axes, deriv_graph, deriv_label).arrange(DOWN, buff=0.1)
        self.place_in_area(deriv_vg, "A1", "C3", scale_factor=0.9)
        
        # Integral Graph: F(x) = 2x (Accumulation)
        int_axes = Axes(x_range=[0, 4], y_range=[0, 8], x_length=2.5, y_length=2.0, 
                        axis_config={"include_tip": False, "color": BLUE_E})
        int_graph = int_axes.plot(lambda x: 2*x, x_range=[0, 3], color=TEAL)
        int_label = Text("Integral (Total)", font_size=18, color=TEAL)
        int_vg = VGroup(int_axes, int_graph, int_label).arrange(DOWN, buff=0.1)
        self.place_in_area(int_vg, "A4", "C6", scale_factor=0.9)

        self.play(Create(deriv_vg), Create(int_vg), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight the slope on the left (#FFFF00) and the area on the right (#00FFFF).
        self.lecture[1].set_color("#FFFF00")
        
        # We'll highlight the 'Slope' concept on the derivative and 'Area' on the integral to show the link.
        slope_highlight = Line(deriv_axes.c2p(1, 2), deriv_axes.c2p(2, 2), color=YELLOW, stroke_width=8)
        slope_label = Text("Slope Value", font_size=16, color=YELLOW)
        # Fix for Issue 38: Move from C2 to A2, scale 0.8
        self.place_at_grid(slope_label, "A2", scale_factor=0.8)

        area_highlight = int_axes.get_area(int_graph, x_range=[0, 2], color="#00FFFF", opacity=0.4)
        area_label = Text("Area", font_size=16, color="#00FFFF")
        self.place_at_grid(area_label, "C5", scale_factor=1.0)

        self.play(Create(slope_highlight), FadeIn(slope_label), FadeIn(area_highlight), FadeIn(area_label), run_time=1.0)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show a climber icon (#FF8C00) on a mountain [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/mountain.svg]; display steepness arrows.
        self.lecture[2].set_color("#FFFF00")
        
        mountain = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mountain.svg").set_color(GREY)
        self.place_in_area(mountain, "D1", "F3", scale_factor=1.2)
        
        climber = Triangle(color="#FF8C00", fill_opacity=1).scale(0.12)
        climber.move_to(self.grid["E2"] + LEFT*0.2 + UP*0.3)
        
        steep_arrow = Arrow(start=ORIGIN, end=UP*0.5+RIGHT*0.5, color="#FF8C00").scale(0.7)
        self.place_at_grid(steep_arrow, "D1", scale_factor=1.0)
        steep_text = Text("Steepness", font_size=14, color="#FF8C00")
        # Fix for Issue 39: Move from D1 to E1, scale 0.8
        self.place_at_grid(steep_text, "E1", scale_factor=0.8)

        self.play(FadeIn(mountain), FadeIn(climber), GrowArrow(steep_arrow), Write(steep_text), run_time=1.0)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Show a growing tree icon (#00FF00) whose height matches the integral area.
        self.lecture[3].set_color("#FFFF00")
        
        trunk = Rectangle(width=0.1, height=0.4, color="#8B4513", fill_opacity=1)
        foliage = Circle(radius=0.3, color="#00FF00", fill_opacity=0.8).move_to(trunk.get_top() + UP*0.2)
        tree = VGroup(trunk, foliage)
        self.place_in_area(tree, "D4", "F6", scale_factor=1.0)
        
        tree_final = tree.copy()
        tree.scale(0.1, about_point=tree.get_bottom())
        
        growth_label = Text("Total Growth", font_size=16, color="#00FF00")
        self.place_at_grid(growth_label, "F5", scale_factor=1.0)

        self.play(Transform(tree, tree_final), Write(growth_label), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Draw a bridge (#FFFFFF) connecting the concepts of 'Slope' and 'Area'.
        self.lecture[4].set_color("#FFFF00")
        
        # Connect the Slope label (now A2) and Area label (C5)
        bridge = Line(self.grid["A2"], self.grid["C5"], color=WHITE, stroke_width=6)
        bridge_label = Text("THE GREAT BRIDGE", font_size=20, color=WHITE)
        # Fix for Issue 37: Move from B2-B5 to C3-C4, scale 0.7
        self.place_in_area(bridge_label, "C3", "C4", scale_factor=0.7)

        self.play(Create(bridge), Write(bridge_label), run_time=1.5)
        self.wait(2)
