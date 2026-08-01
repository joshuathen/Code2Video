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
        # Setup layout with title and lecture lines
        self.setup_layout(
            "Projection and Perspective Shift", 
            [
                "Projections squash 3D objects onto flat 2D planes.", 
                "Top and side views clarify if wires are touching.", 
                "Orthographic views eliminate perspective distortion to simplify problems."
            ]
        )
        
        # Define functions for the projection paths
        def wire_base_func(x):
            return 0.6 * np.sin(1.5 * x) + 0.2 * np.cos(4 * x)

        # Stage 2: Top View (XY Plane)
        # Wire 1: y = base
        # Wire 2: y = base + 0.6
        w1_top = FunctionGraph(lambda x: wire_base_func(x), x_range=[-1.5, 1.5], color="#FFFFFF")
        w2_top = FunctionGraph(lambda x: wire_base_func(x) + 0.6, x_range=[-1.5, 1.5], color="#FFFFFF")
        
        # Stage 3: Side View (XZ Plane)
        # We plot Z against X. Z1 = 0.5, Z2 = -0.5
        w1_side = Line(start=[-1.5, 0.5, 0], end=[1.5, 0.5, 0], color="#FFFFFF")
        w2_side = Line(start=[-1.5, -0.5, 0], end=[1.5, -0.5, 0], color="#FFFFFF")

        # Labels for the views
        view_label = Text("Perspective View", font_size=20, color=WHITE)
        top_label = Text("Top View (XY)", font_size=20, color=WHITE)
        side_label = Text("Side View (XZ)", font_size=20, color=WHITE)

        # Background Plane/Box for context
        visual_box = Rectangle(width=5.5, height=5.0, color=GRAY, stroke_opacity=0.3)
        self.place_in_area(visual_box, 'B1', 'F6')
        self.add(visual_box)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FF00"), run_time=0.5)
        
        # Load asset for tangled wires
        wires_asset = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/wires.svg")
        wires_asset.set_color("#00FF00")
        self.place_in_area(wires_asset, 'B1', 'F6', scale_factor=1.2)
        
        self.place_in_area(view_label, 'A1', 'A6', scale_factor=0.8)
        
        self.play(Create(wires_asset), Write(view_label))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(GRAY),
            self.lecture[1].animate.set_color("#FFFFFF"),
            run_time=0.5
        )
        
        # Prepare Top-down view
        wires_group_2 = VGroup(w1_top, w2_top)
        self.place_in_area(wires_group_2, 'B1', 'F6', scale_factor=0.9)
        
        # Plane graphic
        plane_rect = Rectangle(width=5.0, height=4.5, fill_color=WHITE, fill_opacity=0.1, stroke_width=1)
        self.place_in_area(plane_rect, 'B1', 'F6', scale_factor=0.9)

        # Set up label for next view
        self.place_in_area(top_label, 'A1', 'A6', scale_factor=0.8)

        self.play(
            ReplacementTransform(wires_asset, wires_group_2),
            Transform(view_label, top_label),
            FadeIn(plane_rect)
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(GRAY),
            self.lecture[2].animate.set_color("#FFFFFF"),
            run_time=0.5
        )
        
        # Prepare Side view
        wires_group_3 = VGroup(w1_side, w2_side)
        self.place_in_area(wires_group_3, 'B1', 'F6', scale_factor=0.9)

        # Set up label for side view
        self.place_in_area(side_label, 'A1', 'A6', scale_factor=0.8)

        self.play(
            ReplacementTransform(wires_group_2, wires_group_3),
            Transform(view_label, side_label),
            plane_rect.animate.scale(0.2).set_opacity(0.5) # Simulating rotation of the plane to edge-on
        )
        self.wait(3)
