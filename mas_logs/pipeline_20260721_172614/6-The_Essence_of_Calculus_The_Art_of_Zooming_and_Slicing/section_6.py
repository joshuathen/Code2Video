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
            "Differentiation and integration are opposites, like adding and subtracting.",
            "Finding the slope and finding the area are connected.",
            "One process undoes the other to reveal the whole picture.",
            "This link is the Fundamental Theorem of Calculus.",
            "It unites the two main pillars of the subject."
        ]
        
        self.setup_layout("The Grand Unification: The Inverse Relationship", lecture_lines)

        # Colors
        MACHINE_COLOR = "#A9A9A9"
        DISTANCE_COLOR = "#ADD8E6"
        SPEED_COLOR = "#FF6347"
        HIGHLIGHT_COLOR = "#FFFF00"

        # Helper for mini graphs
        def create_mini_graph(color, label_text):
            axes = Axes(x_range=[0, 3], y_range=[0, 2], x_length=1.5, y_length=1.0, 
                        axis_config={"include_tip": False, "stroke_width": 2}).set_color(GRAY)
            if color == SPEED_COLOR:
                curve = axes.plot(lambda x: 0.5 * x, x_range=[0, 3], color=color)
            else:
                curve = axes.plot(lambda x: 0.1 * x**2, x_range=[0, 3], color=color)
            
            label = Text(label_text, font_size=14, color=color).next_to(axes, UP, buff=0.1)
            return VGroup(axes, curve, label)

        # === Animation for Lecture Line 1 ===
        # Show a 'Math Machine' box (#A9A9A9) with 'Differentiation' and 'Integration' labels.
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        
        machine_box = RoundedRectangle(corner_radius=0.2, width=4, height=3, color=MACHINE_COLOR, fill_opacity=0.2)
        self.place_in_area(machine_box, "B2", "E5")
        
        diff_label = Text("Differentiation", font_size=20, color=WHITE)
        int_label = Text("Integration", font_size=20, color=WHITE)
        
        self.place_at_grid(diff_label, "B3", scale_factor=1.0)
        self.place_at_grid(int_label, "E3", scale_factor=1.0)
        
        self.play(Create(machine_box), Write(diff_label), Write(int_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Finding the slope and finding the area are connected.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_COLOR)
        
        slope_label = Text("(Finding Slope)", font_size=16, color=WHITE).next_to(diff_label, DOWN, buff=0.1)
        area_label = Text("(Finding Area)", font_size=16, color=WHITE).next_to(int_label, UP, buff=0.1)
        
        self.play(FadeIn(slope_label), FadeIn(area_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # A distance graph (#ADD8E6) enters and a speed graph (#FF6347) exits.
        # "One process undoes the other to reveal the whole picture."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)

        dist_graph = create_mini_graph(DISTANCE_COLOR, "Distance")
        speed_graph = create_mini_graph(SPEED_COLOR, "Speed")

        # Entry path for distance graph (Issue 37 Fix: C2, 0.6)
        self.place_at_grid(dist_graph, "C2", scale_factor=0.6)
        self.play(dist_graph.animate.move_to(machine_box.get_center()).set_opacity(0), run_time=1.5)
        
        # Exit path for speed graph (Issue 38 Fix: C5, 0.6)
        speed_graph.move_to(machine_box.get_center()).set_opacity(0)
        self.add(speed_graph)
        speed_graph.generate_target()
        self.place_at_grid(speed_graph.target, "C5", scale_factor=0.6)
        speed_graph.target.set_opacity(1)
        self.play(MoveToTarget(speed_graph), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # This link is the Fundamental Theorem of Calculus.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(HIGHLIGHT_COLOR)
        
        # Issue 39 Fix: A4, 0.9
        ftc_text = Text("Fundamental Theorem\nof Calculus", font_size=24, color=HIGHLIGHT_COLOR, line_spacing=0.8)
        self.place_at_grid(ftc_text, "A4", scale_factor=0.9)
        
        self.play(Write(ftc_text))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The speed graph (#FF6347) enters from the other side and the distance graph exits.
        # It unites the two main pillars of the subject.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(HIGHLIGHT_COLOR)

        # Speed enters from the right (Issue 38 Fix: D5, 0.6)
        speed_graph_rev = create_mini_graph(SPEED_COLOR, "Speed")
        self.place_at_grid(speed_graph_rev, "D5", scale_factor=0.6)
        self.play(speed_graph_rev.animate.move_to(machine_box.get_center()).set_opacity(0), run_time=1.5)
        
        # Distance exits from the left (Issue 37 Fix: D2, 0.6)
        dist_graph_rev = create_mini_graph(DISTANCE_COLOR, "Distance")
        dist_graph_rev.move_to(machine_box.get_center()).set_opacity(0)
        self.add(dist_graph_rev)
        dist_graph_rev.generate_target()
        self.place_at_grid(dist_graph_rev.target, "D2", scale_factor=0.6)
        dist_graph_rev.target.set_opacity(1)
        self.play(MoveToTarget(dist_graph_rev), run_time=1.5)
        
        self.wait(2)
