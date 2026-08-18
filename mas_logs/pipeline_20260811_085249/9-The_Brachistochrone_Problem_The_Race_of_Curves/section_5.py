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
        lecture_lines = ["Calculus of Variations found the solution.", "Time is independent of start point.", "Applied in efficient modern engineering."]
        self.setup_layout("Summary & Real-world Application", lecture_lines)
        
        # Assets
        pendulum = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pendulum.svg")
        bridge = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bridge.svg")
        
        # Elements
        cycloid_path = ParametricFunction(
            lambda t: np.array([t - np.sin(t), - (1 - np.cos(t)), 0]),
            t_range=[0, 2*PI], color=WHITE
        )
        
        label_cycloid = Text("Cycloid Path", font_size=24, color=WHITE)
        t_label = Text("Tautochrone", font_size=24, color="#3498DB")

        # === Animation for Lecture Line 1 ===
        # Summarize key concepts: Brachistochrone = fastest path. [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/pendulum.svg] (Color: #FFFFFF)
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.place_at_grid(cycloid_path, 'D2', scale_factor=0.6)
        self.place_at_grid(pendulum, 'B2', scale_factor=0.5)
        self.add(cycloid_path, pendulum)

        # === Animation for Lecture Line 2 ===
        # Display examples of the cycloid in nature/engineering. (Color: #3498DB)
        self.play(self.lecture[1].animate.set_color("#3498DB"))
        self.place_at_grid(t_label, 'D4', scale_factor=0.7)
        self.play(Write(t_label))

        # === Animation for Lecture Line 3 ===
        # Final screen showing the path label 'Cycloid'. [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/bridge.svg] (Color: #2ECC71)
        self.play(self.lecture[2].animate.set_color("#2ECC71"))
        self.place_at_grid(label_cycloid, 'F2', scale_factor=0.7)
        self.place_at_grid(bridge, 'F5', scale_factor=0.5)
        self.play(Write(label_cycloid), FadeIn(bridge))
        self.wait(2)
