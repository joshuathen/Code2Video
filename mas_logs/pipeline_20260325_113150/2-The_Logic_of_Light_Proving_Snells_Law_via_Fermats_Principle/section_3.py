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
        # Setup layout with mandatory title and lecture lines
        lecture_lines = [
            "Consider light traveling between points A and B.",
            "Points A and B are at heights h1 and h2.",
            "The total horizontal distance is L, meeting at point x.",
            "The path creates angles theta 1 and theta 2.",
            "This splits the interface into segments x and L minus x."
        ]
        self.setup_layout("Setting the Stage: The Geometry of Refraction", lecture_lines)

        # Color definitions
        color1 = YELLOW
        color2 = GREEN
        color3 = ORANGE
        color4 = RED
        color5 = "#ADD8E6"  # Light Blue

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(color1))
        
        interface = Line(self.grid["C1"], self.grid["C6"], color=WHITE)
        dotA = Dot(color=color1)
        dotB = Dot(color=color1)
        self.place_at_grid(dotA, "B1")
        self.place_at_grid(dotB, "E5")
        
        labA = Text("A", color=color1, font_size=24)
        labB = Text("B", color=color1, font_size=24)
        self.place_at_grid(labA, "A1")
        self.place_at_grid(labB, "F5")
        
        self.play(Create(interface))
        self.play(FadeIn(dotA), FadeIn(dotB), Write(labA), Write(labB))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(color2))
        
        h1_line = DashedLine(self.grid["B1"], self.grid["C1"], color=color2)
        h2_line = DashedLine(self.grid["E5"], self.grid["C5"], color=color2)
        
        labh1 = Text("h1", color=color2, font_size=24)
        labh2 = Text("h2", color=color2, font_size=24)
        self.place_at_grid(labh1, "A2")
        self.place_at_grid(labh2, "F6")
        
        self.play(Create(h1_line), Create(h2_line))
        self.play(Write(labh1), Write(labh2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(color3))
        
        brace_l = BraceBetweenPoints(self.grid["C1"], self.grid["C5"], direction=DOWN, color=color3)
        labL = Text("L", color=color3, font_size=24)
        # Issue 38 Fix: Place labL at F3, scale 0.8
        self.place_at_grid(labL, "F3", scale_factor=0.8)
        
        dotX = Dot(color=color3)
        self.place_at_grid(dotX, "C3")
        
        self.play(Create(brace_l), Write(labL))
        self.play(FadeIn(dotX))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(color4))
        
        ray1 = Line(self.grid["B1"], self.grid["C3"], color=color4)
        ray2 = Line(self.grid["C3"], self.grid["E5"], color=color4)
        
        normal = DashedLine(self.grid["B3"], self.grid["D3"], color=GRAY)
        
        l1 = Line(self.grid["C3"], self.grid["B3"])
        l2 = Line(self.grid["C3"], self.grid["B1"])
        angle1 = Angle(l1, l2, radius=0.3, color=color4)
        
        lab_theta1 = Text("θ1", color=color4, font_size=24)
        # Issue 37 Fix: Place lab_theta1 at C2, scale 0.6
        self.place_at_grid(lab_theta1, "C2", scale_factor=0.6)
        
        l3 = Line(self.grid["C3"], self.grid["D3"])
        l4 = Line(self.grid["C3"], self.grid["E5"])
        angle2 = Angle(l3, l4, radius=0.3, color=color4)
        
        lab_theta2 = Text("θ2", color=color4, font_size=24)
        self.place_at_grid(lab_theta2, "D4", scale_factor=0.6)
        
        self.play(Create(ray1), Create(ray2))
        self.play(Create(normal))
        self.play(Create(angle1), Write(lab_theta1), Create(angle2), Write(lab_theta2))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(color5))
        
        brace_x = BraceBetweenPoints(self.grid["C1"], self.grid["C3"], direction=UP, color=color5)
        labX = Text("x", color=color5, font_size=24)
        self.place_at_grid(labX, "B2")
        
        brace_lx = BraceBetweenPoints(self.grid["C3"], self.grid["C5"], direction=UP, color=color5)
        labLX = Text("L-x", color=color5, font_size=24)
        # Issue 39 Fix: Place labLX in area A4-B4, scale 0.7
        self.place_in_area(labLX, "A4", "B4", scale_factor=0.7)
        
        self.play(Create(brace_x), Write(labX))
        self.play(Create(brace_lx), Write(labLX))
        self.wait(2)
