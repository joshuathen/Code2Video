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
        self.setup_layout("Summary & Synthesis", ["Step one: Choose a kernel.", "Step two: Slide, multiply, and sum.", "Step three: Analyze the resulting map."])
        
        # Load SVG Assets
        input_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg", color="#FFFFFF")
        kernel_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sensor.svg", color="#FF0000")
        
        input_label = Text("Input", color="#FFFFFF")
        kernel_label = Text("Kernel", color="#FF0000")
        feature_map_label = Text("Feature Map", color="#00FF00")
        
        # Positioning
        # B004/B033: Grid 4-6 for main objects, restrict to C4-D6 for safety.
        # Placing assets and labels
        self.place_at_grid(input_asset, "B4", scale_factor=0.6)
        self.place_at_grid(input_label, "B5", scale_factor=0.7) # B011: Tethered
        input_label.next_to(input_asset, DOWN, buff=0.1)
        
        self.place_at_grid(kernel_asset, "C4", scale_factor=0.6)
        self.place_at_grid(kernel_label, "C5", scale_factor=0.7) # B011
        kernel_label.next_to(kernel_asset, DOWN, buff=0.1)
        
        self.place_at_grid(feature_map_label, "E4", scale_factor=0.7) # B034
        
        # Arrows (B035)
        arrow_in_to_ker = Arrow(start=input_asset.get_bottom(), end=kernel_asset.get_top(), color=WHITE, buff=0.1)
        arrow_ker_to_feat = Arrow(start=kernel_asset.get_bottom(), end=feature_map_label.get_top(), color=WHITE, buff=0.1) # B035
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(input_asset), Write(input_label))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(kernel_asset), Write(kernel_label), GrowArrow(arrow_in_to_ker))
        self.lecture[1].set_color("#FF0000")
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.play(Write(feature_map_label), GrowArrow(arrow_ker_to_feat))
        self.lecture[2].set_color("#00FF00")
        self.wait(1)
