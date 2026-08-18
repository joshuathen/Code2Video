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
        self.setup_layout("Prerequisites: Tangent Lines", 
                          ["Derivatives represent the slope of a curve.", 
                           "Tangent lines approximate a curve's local behavior.", 
                           "We use them to find root approximations."])
        
        # Define function and point
        axes = Axes(x_range=[-1, 5, 1], y_range=[-1, 5, 1], axis_config={"include_tip": False})
        self.place_in_area(axes, 'C2', 'F6', scale_factor=0.5)
        
        func = axes.plot(lambda x: 0.2*(x-2)**3 + 1, color=WHITE)
        a = 3.5
        point_a = Dot(axes.c2p(a, 0.2*(a-2)**3 + 1), color="#FF00FF")
        
        # === Animation for Lecture Line 1 ===
        self.play(Create(func), run_time=1)
        self.lecture[0].set_color("#FF00FF")
        self.play(FadeIn(point_a), run_time=0.5)

        # === Animation for Lecture Line 2 ===
        slope = 0.6*(a-2)**2
        tangent = TangentLine(func, alpha=0.5, length=4, color="#00FFFF")
        tangent.move_to(axes.c2p(a, 0.2*(a-2)**3 + 1))
        
        self.lecture[1].set_color("#00FFFF")
        self.play(Create(tangent), run_time=1)

        # === Animation for Lecture Line 3 ===
        root_val = a - (0.2*(a-2)**3 + 1) / slope
        root_point = Dot(axes.c2p(root_val, 0), color="#FFFF00")
        x1_label = Text("x_1", font_size=20, color="#FFFF00")
        self.place_at_grid(x1_label, 'E2', scale_factor=0.6)
        
        # Using asset
        try:
            asset_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
            asset_img.scale(0.3).next_to(x1_label, RIGHT)
            self.add(asset_img)
        except Exception as e:
            pass
            
        self.lecture[2].set_color("#FFFF00")
        self.play(Create(root_point), Write(x1_label), run_time=1)
        self.wait(1)
