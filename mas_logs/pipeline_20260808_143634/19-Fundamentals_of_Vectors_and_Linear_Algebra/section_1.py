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
        self.setup_layout("Introduction: What is a Vector?", [
            "Vectors have both magnitude and direction.",
            "Scalars possess only magnitude, like numbers.",
            "Vectors are represented in a coordinate plane."
        ])
        
        # Asset Loading
        plane_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/plane.svg"
        plane_icon = SVGMobject(plane_asset)
        
        # Setup initial components
        origin_pos = self.grid["F2"]
        end_pos = self.grid["C4"]
        
        vector = Arrow(start=origin_pos, end=end_pos, color=WHITE, buff=0)
        label_v = Text("v", color=WHITE, font_size=24).next_to(vector.get_center(), UP)
        
        # Group for VideoCritic 20/21
        vector_group = VGroup(vector, label_v)
        self.place_at_grid(plane_icon, 'B3', scale_factor=0.6) # Using grid for plane
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(vector), Write(label_v), FadeIn(plane_icon))
        self.lecture[0].set_color("#FFD700") 
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(vector.animate.set_color("#FF0000"))
        self.lecture[1].set_color("#FF0000")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        h_line = DashedLine(start=origin_pos, end=np.array([end_pos[0], origin_pos[1], 0]), color="#00FF00")
        v_line = DashedLine(start=np.array([end_pos[0], origin_pos[1], 0]), end=end_pos, color="#00FF00")
        label_x = Text("x", color="#00FF00", font_size=20)
        label_y = Text("y", color="#00FF00", font_size=20)
        
        self.place_at_grid(label_x, 'B2', scale_factor=0.5)
        self.place_at_grid(label_y, 'C4', scale_factor=0.5)
        
        # Coordinate points (VideoCritic 22)
        point_x = Dot(color="#FFFF00")
        point_y = Dot(color="#FFFF00")
        self.place_at_grid(point_x, 'E2', scale_factor=0.4)
        self.place_at_grid(point_y, 'E5', scale_factor=0.4)
        
        self.play(Create(h_line), Create(v_line), FadeIn(label_x), FadeIn(label_y), FadeIn(point_x), FadeIn(point_y))
        self.lecture[2].set_color("#00FF00")
        self.wait(1)
        
        # Final position move
        new_end = self.grid["B5"]
        self.play(
            vector.animate.put_start_and_end_on(origin_pos, new_end),
            label_v.animate.next_to(new_end, UP),
            plane_icon.animate.move_to(new_end)
        )
        
        # Flash components
        self.play(Flash(h_line.get_center(), color="#FFFF00"), Flash(v_line.get_center(), color="#FFFF00"))
        self.wait(1)
