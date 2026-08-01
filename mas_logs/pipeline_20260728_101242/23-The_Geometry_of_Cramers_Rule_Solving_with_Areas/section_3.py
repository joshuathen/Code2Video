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

class Section3Scene(TeachingScene):
    def construct(self):
        # Setup title and lecture lines
        title_text = "The Geometric Transformation"
        lecture_lines = [
            "Target b is a sum of scaled vectors.",
            "We scale v1 by x and v2 by y.",
            "Their sum reaches the drone's destination exactly."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Define colors
        v1_color = BLUE_B
        v2_color = GREEN_B
        b_color = "#FF0000"  # Mandatory red for target b
        scaling_color = YELLOW_B

        # 1. Coordinate System
        # Issue 26 Fix: Use scale_factor=0.85 to avoid labels hitting boundaries
        plane = NumberPlane(
            x_range=[0, 8, 1],
            y_range=[0, 8, 1],
            x_length=4.5,
            y_length=4.5,
            axis_config={"include_tip": True, "color": GREY},
            background_line_style={"stroke_opacity": 0.3}
        )
        self.place_in_area(plane, "A1", "F6", scale_factor=0.85)
        self.add(plane)

        # Vector data
        v1_coords = np.array([3, 2, 0])
        v2_coords = np.array([1, 2, 0])
        b_coords = np.array([7, 6, 0])
        
        # Base vectors
        v1 = Arrow(plane.coords_to_point(0,0), plane.coords_to_point(*v1_coords), buff=0, color=v1_color, stroke_width=4)
        v2 = Arrow(plane.coords_to_point(0,0), plane.coords_to_point(*v2_coords), buff=0, color=v2_color, stroke_width=4)
        
        v1_label = Text("v1", font_size=18, color=v1_color, slant=ITALIC).next_to(v1.get_end(), DR, buff=0.1)
        v2_label = Text("v2", font_size=18, color=v2_color, slant=ITALIC).next_to(v2.get_end(), UL, buff=0.1)

        self.add(v1, v2, v1_label, v2_label)
        self.wait(1)

        # === Animation for Lecture Line 1 ===
        # "Target b is a sum of scaled vectors."
        self.play(self.lecture[0].animate.set_color(b_color))
        
        b_vec = Arrow(plane.coords_to_point(0,0), plane.coords_to_point(*b_coords), buff=0, color=b_color, stroke_width=5)
        b_label = Text("b", font_size=20, color=b_color, weight=BOLD)
        
        # Issue 28 Fix: Position label 'b' at grid 'A6' to avoid cramping
        self.place_at_grid(b_label, "A6", scale_factor=0.7)
        
        self.play(Create(b_vec), FadeIn(b_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "We scale v1 by x and v2 by y."
        # Calculation: 2*v1 + 1*v2 = 2*[3,2] + 1*[1,2] = [6,4] + [1,2] = [7,6] = b
        self.play(self.lecture[1].animate.set_color(scaling_color))
        
        # Scaling v1 to 2*v1
        xv1_coords = 2 * v1_coords
        xv1_dashed = DashedLine(plane.coords_to_point(0,0), plane.coords_to_point(*xv1_coords), color=scaling_color, dash_length=0.1)
        xv1_label = Text("x v1", font_size=18, color=scaling_color).next_to(xv1_dashed.get_end(), DOWN, buff=0.1)
        
        # Scaling v2 to 1*v2
        yv2_dashed = DashedLine(plane.coords_to_point(0,0), plane.coords_to_point(*v2_coords), color=scaling_color, dash_length=0.1)
        yv2_label = Text("y v2", font_size=18, color=scaling_color).next_to(yv2_dashed.get_end(), LEFT, buff=0.1)

        self.play(Create(xv1_dashed), FadeIn(xv1_label))
        self.play(Create(yv2_dashed), FadeIn(yv2_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Their sum reaches the drone's destination exactly."
        self.play(self.lecture[2].animate.set_color(b_color))
        
        # Issue 20 & 27: Asset integration and correct scaling
        drone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/drone.svg")
        drone.scale(0.4) # Applying scale factor from Issue 27
        drone.move_to(plane.coords_to_point(0,0))
        
        # Vector addition visual
        yv2_added_dashed = DashedLine(plane.coords_to_point(*xv1_coords), plane.coords_to_point(*b_coords), color=scaling_color, dash_length=0.1)
        
        self.play(FadeIn(drone))
        # Drone follows the parallelogram path
        self.play(drone.animate.move_to(plane.coords_to_point(*xv1_coords)), run_time=1.5)
        
        self.play(
            yv2_dashed.animate.move_to(yv2_added_dashed.get_center()),
            yv2_label.animate.next_to(plane.coords_to_point(*b_coords), RIGHT, buff=0.1),
            # Issue 27 Fix: Final drone position at 'B6' to avoid obscuring vector head
            drone.animate.move_to(self.grid["B6"]),
            run_time=2
        )
        
        # Final highlight
        self.play(Indicate(b_vec, color=WHITE))
        self.play(b_vec.animate.set_stroke(width=8))
        self.wait(2)
