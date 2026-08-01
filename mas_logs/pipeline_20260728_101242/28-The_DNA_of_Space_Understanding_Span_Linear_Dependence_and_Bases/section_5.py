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
        # Initialize Scene
        lecture_lines = [
            "Three independent vectors span our 3D world.",
            "Dependence traps movement within a lower-dimensional slice.",
            "These concepts form the DNA of all spatial math."
        ]
        self.setup_layout("Summary: Building 3D Worlds", lecture_lines)

        # Colors
        COLOR_X = "#FF0000"
        COLOR_Y = "#00FF00"
        COLOR_Z = "#0000FF"
        COLOR_HIGHLIGHT = "#FFFF00"
        COLOR_SUMMARY = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Three independent vectors span our 3D world.
        self.lecture[0].set_color(YELLOW)
        
        # Origin at D2
        origin = self.grid["D2"]
        
        # Axes - simulated 3D
        axis_x = Arrow(origin, origin + RIGHT * 2, color=COLOR_X, buff=0)
        axis_y = Arrow(origin, origin + UP * 2, color=COLOR_Y, buff=0)
        axis_z = Arrow(origin, origin + np.array([1, 1, 0]), color=COLOR_Z, buff=0)
        
        label_x = Text("X", font_size=18, color=COLOR_X).next_to(axis_x, RIGHT, buff=0.1)
        label_y = Text("Y", font_size=18, color=COLOR_Y).next_to(axis_y, UP, buff=0.1)
        label_z = Text("Z", font_size=18, color=COLOR_Z).next_to(axis_z, UR, buff=0.1)
        
        # Vectors - slightly offset to see independence clearly
        v1 = Arrow(origin, origin + RIGHT * 1.5 + UP * 0.2, color=RED_A, buff=0)
        v2 = Arrow(origin, origin + UP * 1.5 + RIGHT * 0.1, color=GREEN_A, buff=0)
        v3 = Arrow(origin, origin + np.array([0.7, 0.7, 0]) * 1.2, color=BLUE_A, buff=0)
        
        # Volume (Polygons to simulate a 3D box)
        p_xy = Polygon(origin, origin + RIGHT * 1.5, origin + RIGHT * 1.5 + UP * 1.5, origin + UP * 1.5, 
                      fill_opacity=0.3, fill_color=COLOR_HIGHLIGHT, stroke_width=0)
        p_xz = Polygon(origin, origin + RIGHT * 1.5, origin + RIGHT * 1.5 + np.array([0.8, 0.8, 0]), origin + np.array([0.8, 0.8, 0]),
                      fill_opacity=0.2, fill_color=COLOR_HIGHLIGHT, stroke_width=0)
        p_yz = Polygon(origin, origin + UP * 1.5, origin + UP * 1.5 + np.array([0.8, 0.8, 0]), origin + np.array([0.8, 0.8, 0]),
                      fill_opacity=0.2, fill_color=COLOR_HIGHLIGHT, stroke_width=0)
        
        volume = VGroup(p_xy, p_xz, p_yz)

        self.play(Create(axis_x), Create(axis_y), Create(axis_z), Write(label_x), Write(label_y), Write(label_z))
        self.play(Create(v1), Create(v2), Create(v3))
        self.play(FadeIn(volume))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Dependence traps movement within a lower-dimensional slice.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Collapse v3 into the XY plane
        v3_target_end = origin + RIGHT * 0.8 + UP * 0.8
        
        # Drone representation - Issue 17: Use SVGMobject
        drone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/drone.svg", color=WHITE)
        drone_label = Text("Drone", font_size=16).next_to(drone, UP, buff=0.1)
        drone_group = VGroup(drone, drone_label)
        
        # Issue 28: Fix position to C5
        self.place_at_grid(drone_group, "C5", scale_factor=0.7)

        self.play(
            v3.animate.put_start_and_end_on(origin, v3_target_end),
            volume[1].animate.scale(0, about_point=origin),
            volume[2].animate.scale(0, about_point=origin),
            volume[0].animate.set_opacity(0.6).set_color(RED),
            run_time=2
        )
        
        # Drone movement - drone moves from outside and "hits" the boundary of the plane
        target_drone_pos = self.grid["C3"]
        self.play(drone_group.animate.move_to(target_drone_pos), run_time=1.5)
        
        # Collision effect
        self.play(Flash(target_drone_pos, color=RED, flash_radius=0.3), drone_group.animate.shift(LEFT*0.1), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # These concepts form the DNA of all spatial math.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        summary_text = Text("3 Independent Vectors = 3D Basis", font_size=24, color=COLOR_SUMMARY)
        # Issue 29: Use place_in_area
        self.place_in_area(summary_text, "F2", "F5", scale_factor=0.9)
        
        self.play(Write(summary_text))
        self.wait(3)
