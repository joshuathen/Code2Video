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
        lecture_lines = [
            "Use Dual Coding Theory.", 
            "Pair verbal with spatial representation.", 
            "Visuals make abstract math move.", 
            "Show relationships between variables clearly.", 
            "Dynamic graphs enhance deep understanding."
        ]
        self.setup_layout("Criterion 2: Visual-Spatial Encoding", lecture_lines)
        
        # Define elements
        node1 = Dot(color=BLUE)
        node2 = Dot(color=BLUE)
        node3 = Dot(color=BLUE)
        
        label1 = Text("Node 1", font_size=20, color=WHITE)
        label2 = Text("Node 2", font_size=20, color=WHITE)
        label3 = Text("Node 3", font_size=20, color=WHITE)
        
        # Grid placement per critic feedback
        self.place_in_area(node1, 'B2', 'B3', scale_factor=0.9)
        self.place_in_area(node2, 'B4', 'B5', scale_factor=0.9)
        self.place_at_grid(node3, 'D4', scale_factor=0.9)
        
        # Labels 1 unit below nodes
        self.place_at_grid(label1, 'C2', scale_factor=0.7)
        self.place_at_grid(label2, 'C4', scale_factor=0.7)
        self.place_at_grid(label3, 'E4', scale_factor=0.7)
        
        line1 = Line(node1.get_center(), node2.get_center(), color=WHITE)
        line2 = Line(node2.get_center(), node3.get_center(), color=WHITE)
        connections = VGroup(line1, line2, node1, node2, node3, label1, label2, label3)
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg
        # Note: SVG might fail if path doesn't exist; ensuring robustness
        try:
            icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
            self.place_at_grid(icon, 'A2', scale_factor=0.5)
        except:
            icon = Dot(color=RED).scale(0.5)
            self.place_at_grid(icon, 'A2')

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.play(FadeIn(icon))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF4500"))
        self.play(Create(connections))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF4500"))
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FF4500"))
        self.play(Indicate(connections))
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FF4500"))
        self.wait(1)
