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
        self.setup_layout("Synthesis: The Fractal Geometry of Computation", 
                          ["Hanoi moves map to Sierpinski points.", 
                           "Ternary addresses define fractal coordinates.", 
                           "Infinite recursion creates the geometric limit."])
        
        # Asset path
        DISK_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg"
        
        # 3-disk Hanoi state diagram (using assets)
        nodes = [SVGMobject(DISK_ASSET, color=YELLOW) for _ in range(7)]
        for i, node in enumerate(nodes):
            self.place_at_grid(node, f"{['B2', 'B4', 'C1', 'C3', 'C5', 'D2', 'D4'][i]}", scale_factor=0.2)
            
        edges = VGroup(*[Line(nodes[i].get_center(), nodes[j].get_center(), color=YELLOW) 
                        for i, j in [(0, 2), (0, 3), (2, 5), (3, 5), (1, 3), (1, 4), (3, 6), (4, 6)]])
        
        hanoi_graph = VGroup(*nodes, edges)
        
        # 3rd iteration Sierpinski triangle
        def get_sierpinski_triangle(order=2):
            if order == 0:
                return Triangle(color=BLUE_C)
            else:
                prev = get_sierpinski_triangle(order-1)
                t1 = prev.copy()
                t2 = prev.copy()
                t3 = prev.copy()
                VGroup(t1, t2, t3).arrange(UP, buff=0)
                return VGroup(t1, t2, t3)
        
        sierpinski = get_sierpinski_triangle(2).set_color(BLUE_C)
        # Apply the fix suggested by VideoCritic (Issue 31/32)
        self.place_in_area(sierpinski, 'C4', 'F6', scale_factor=0.45)
        sierpinski.set_opacity(0)
        
        # === Animation for Lecture Line 1 ===
        self.play(Create(hanoi_graph))
        self.lecture[0].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(sierpinski))
        self.lecture[1].set_color(BLUE_C)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(Transform(hanoi_graph, sierpinski))
        self.play(FadeOut(hanoi_graph))
        self.lecture[2].set_color(WHITE)
        self.wait(2)
