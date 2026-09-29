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
        self.setup_layout("Application: The Geometry of a Bouncing Ball", 
                          ["We apply this to complex paths.", 
                           "Find tangent slopes for any point.", 
                           "Easily determine velocity on any curve."])
        
        # --- Elements ---
        equation = MathTex("x^2 + y^2 = r^2", color="#FFFFFF")
        circle = Circle(radius=0.8, color="#00FF00")
        ball = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg", color="#00FF00")
        tangent_line = Line(start=LEFT*0.5, end=RIGHT*0.5, color="#FFD700")
        
        # --- Positions ---
        self.place_in_area(equation, 'A2', 'B4', scale_factor=1.0)
        self.place_at_grid(circle, 'D2', scale_factor=0.8)
        
        # --- Animation ---
        # === Animation for Lecture Line 1 ===
        self.play(Write(equation))
        self.lecture[0].set_color("#FFFFFF")
        
        # === Animation for Lecture Line 2 ===
        self.play(Create(circle))
        ball.scale(0.3)
        self.add(ball)
        
        def update_ball(mob):
            angle = self.time * 2
            mob.move_to(circle.point_at_angle(angle))
            
        ball.add_updater(update_ball)
        self.lecture[1].set_color("#00FF00")
        self.wait(2)
        
        # === Animation for Lecture Line 3 ===
        self.place_at_grid(tangent_line, 'D4', scale_factor=0.8)
        self.add(tangent_line)
        
        def update_tangent(mob):
            angle = self.time * 2
            point = circle.point_at_angle(angle)
            mob.move_to(point)
            # Tangent is perpendicular to radius vector (angle + PI/2)
            mob.set_angle(angle + PI/2)
        
        tangent_line.add_updater(update_tangent)
        self.lecture[2].set_color("#FFD700")
        self.wait(3)
        
        # Cleanup
        ball.remove_updater(update_ball)
        tangent_line.remove_updater(update_tangent)
