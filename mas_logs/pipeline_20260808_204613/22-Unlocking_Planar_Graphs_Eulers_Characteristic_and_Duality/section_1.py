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
        lecture_lines = [
            "Planar graphs draw in 2D with no edges crossing.",
            "A 'face' is any enclosed region, including the exterior.",
            "Visualize a house: five vertices, six edges.",
            "The house has two interior and one exterior face.",
            "Three faces total: satisfy the planarity condition."
        ]
        self.setup_layout("Prerequisites: Visualizing Planarity", lecture_lines)
        
        # House asset
        house_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/house.svg", color=BLUE)
        self.place_in_area(house_icon, "B1", "E4", scale_factor=0.9)
        
        # Define house nodes
        nodes = {
            "v1": Dot(color=BLUE), # Bottom Left
            "v2": Dot(color=BLUE), # Bottom Right
            "v3": Dot(color=BLUE), # Top Right
            "v4": Dot(color=BLUE), # Top Left
            "v5": Dot(color=BLUE)  # Roof Peak
        }
        
        # Labels
        labels = {
            "v1": Text("A", font_size=20),
            "v2": Text("B", font_size=20),
            "v3": Text("C", font_size=20),
            "v4": Text("D", font_size=20),
            "v5": Text("E", font_size=20)
        }
        
        # Position vertices around house_icon (approximate coords)
        # Using the grid logic conceptually
        nodes["v1"].move_to(house_icon.get_corner(DL))
        nodes["v2"].move_to(house_icon.get_corner(DR))
        nodes["v3"].move_to(house_icon.get_right() + UP*0.5)
        nodes["v4"].move_to(house_icon.get_left() + UP*0.5)
        nodes["v5"].move_to(house_icon.get_top())
        
        # Fix: Place specific labels based on feedback
        self.place_at_grid(labels["v1"], "D1", scale_factor=0.6)
        self.place_at_grid(labels["v5"], "A2", scale_factor=0.6)
        
        edges = [
            Line(nodes["v1"].get_center(), nodes["v2"].get_center(), color=YELLOW),
            Line(nodes["v2"].get_center(), nodes["v3"].get_center(), color=YELLOW),
            Line(nodes["v3"].get_center(), nodes["v4"].get_center(), color=YELLOW),
            Line(nodes["v4"].get_center(), nodes["v1"].get_center(), color=YELLOW),
            Line(nodes["v4"].get_center(), nodes["v5"].get_center(), color=YELLOW),
            Line(nodes["v5"].get_center(), nodes["v3"].get_center(), color=YELLOW),
        ]
        
        graph = VGroup(house_icon, *nodes.values(), *edges, *labels.values())

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE), Write(graph))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(ORANGE))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(RED))
        self.wait(2)
