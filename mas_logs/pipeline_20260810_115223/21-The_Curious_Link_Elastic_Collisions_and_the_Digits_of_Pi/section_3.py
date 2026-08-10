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
        self.setup_layout("The Mathematical Mapping", [
            "Mass ratio 100^n gives n+1 digits of Pi.",
            "System phase space maps to circular motion.",
            "Collisions act as a digits calculator.",
            "Pi emerges from simple physical rules.",
            "Visual representation of the correlation."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Display M = 100^n * m
        eq = MathTex("M = 100^n \\cdot m", color="#00BFFF")
        self.place_at_grid(eq, 'B2', scale_factor=1.1)
        # Using asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg
        blocks = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg", color="#00BFFF")
        self.place_at_grid(blocks, 'B4', scale_factor=0.5)
        self.play(Write(eq), FadeIn(blocks))
        self.lecture[0].set_color("#00BFFF")

        # === Animation for Lecture Line 2 ===
        # Display 31 and Pi
        num_31 = Text("31", color="#FFD700", font_size=48)
        pi_approx = MathTex("\\pi \\approx 3.14159...", color=WHITE)
        self.place_at_grid(num_31, 'C3', scale_factor=1.0)
        self.place_at_grid(pi_approx, 'C5', scale_factor=0.9)
        self.play(FadeIn(num_31), FadeIn(pi_approx))
        self.lecture[1].set_color("#FFD700")

        # === Animation for Lecture Line 3 ===
        # Visualize a circle
        circle = Circle(radius=1.5, color=WHITE)
        line = Line(start=ORIGIN, end=UP*1.5, color=RED)
        phase_space = VGroup(circle, line)
        # Using asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/billiards.svg
        billiards = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/billiards.svg", color=WHITE)
        billiards_grp = VGroup(phase_space, billiards).arrange(RIGHT)
        
        self.place_in_area(billiards_grp, 'D3', 'E5', scale_factor=0.55)
        self.play(Create(circle), Create(line), FadeIn(billiards))
        self.play(Rotate(line, angle=2*PI, about_point=circle.get_center(), run_time=2))
        self.lecture[2].set_color(RED)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(GREEN)
        self.wait(1)
        
        # === Animation for Lecture Line 5 ===
        # Asset: PiCollisionGraph (replaced with dummy object as file path is missing)
        pi_graph = Text("PiCollisionGraph", color=GRAY)
        self.place_at_grid(pi_graph, 'F3', scale_factor=0.5)
        self.play(FadeIn(pi_graph))
        self.lecture[4].set_color(GRAY)
        self.wait(2)
