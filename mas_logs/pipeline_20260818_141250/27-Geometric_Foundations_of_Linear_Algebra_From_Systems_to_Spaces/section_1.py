from manim import *
import os

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
            "Linear systems are intersections of geometric objects.",
            "Equations define lines or planes in space.",
            "The solution is where these objects meet.",
            "Robots A and B track these specific paths.",
            "Their collision point is the unique solution."
        ]
        self.setup_layout("Geometric Interpretation of Linear Systems", lecture_lines)
        
        # Paths for robots
        line1 = Line(start=self.grid["B1"], end=self.grid["E6"], color=WHITE)
        line2 = Line(start=self.grid["E1"], end=self.grid["B6"], color=WHITE)
        intersecting_lines = VGroup(line1, line2)
        self.place_in_area(intersecting_lines, 'B4', 'E6', scale_factor=0.9)
        
        # === Animation for Lecture Line 1 ===
        self.play(Create(line1), Create(line2))
        self.lecture[0].set_color("#FFFFFF")
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight intersection point as solution
        collision_dot = Dot(self.grid["C4"], color=YELLOW, radius=0.1)
        self.place_at_grid(collision_dot, 'C4', scale_factor=0.5)
        label = Text('Solution', font_size=24).next_to(collision_dot, UP)
        self.play(FadeIn(collision_dot), Write(label))
        self.lecture[2].set_color("#FFFF00")
        
        # === Animation for Lecture Line 4 ===
        # Robots A and B
        robotA = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg").set_color(GREEN)
        robotB = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg").set_color(GREEN)
        self.place_at_grid(robotA, 'B1', scale_factor=0.3)
        self.place_at_grid(robotB, 'E1', scale_factor=0.3)
        self.play(FadeIn(robotA), FadeIn(robotB))
        self.lecture[3].set_color("#00FF00")
        
        # === Animation for Lecture Line 5 ===
        # Collision
        self.play(
            robotA.animate.move_to(collision_dot.get_center()),
            robotB.animate.move_to(collision_dot.get_center()),
            run_time=2
        )
        self.play(Flash(collision_dot, color=RED, line_length=0.2, num_lines=12))
        self.lecture[4].set_color("#FF0000")
        self.wait(2)
