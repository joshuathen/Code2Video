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
        self.setup_layout("Introduction: The Game as a Decision Tree", [
            "Wordle is a search problem.",
            "It uses a finite set of solutions.",
            "Information gain reduces remaining possibilities.",
            "Each guess acts as a filter.",
            "Visualizing the branching decision tree."
        ])
        
        # Create elements
        root = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/game.svg").set_color(WHITE)
        node1 = Circle(radius=0.2, color="#FF00FF", fill_opacity=1)
        node2 = Circle(radius=0.2, color="#FF00FF", fill_opacity=1)
        
        label1 = Text("Outcome 1", font_size=12, color="#00FFFF")
        label2 = Text("Outcome 2", font_size=12, color="#00FFFF")
        
        # We need the positions to draw the lines
        # root is at D4. node1 at E3, node2 at E5
        # We'll calculate the lines after placing
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        self.place_at_grid(root, 'D4', scale_factor=0.5)
        self.play(FadeIn(root))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF00FF")
        self.place_at_grid(node1, 'E3', scale_factor=0.8)
        self.place_at_grid(node2, 'E5', scale_factor=0.8)
        line1 = Line(root.get_bottom(), node1.get_top(), color="#FF00FF")
        line2 = Line(root.get_bottom(), node2.get_top(), color="#FF00FF")
        self.play(FadeIn(node1), FadeIn(node2), Create(line1), Create(line2))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FFFF")
        self.place_at_grid(label1, 'F3', scale_factor=0.7)
        self.place_at_grid(label2, 'F5', scale_factor=0.7)
        self.play(Write(label1), Write(label2))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFFF00")
        path = Line(root.get_bottom(), node1.get_top(), color="#FFFF00", stroke_width=4)
        self.play(Create(path))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#00FF00")
        final_node = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg").set_color("#00FF00")
        self.place_at_grid(final_node, 'E3', scale_factor=0.5)
        self.play(ReplacementTransform(node1, final_node))
        self.wait(1)
