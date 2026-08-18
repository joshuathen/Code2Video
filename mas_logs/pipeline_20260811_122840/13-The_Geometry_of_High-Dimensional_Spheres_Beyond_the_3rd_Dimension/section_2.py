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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Curse of Dimensionality: Volume Concentration", [
            "Volume hides near the outer shell.", 
            "Imagine spheres as nested onion layers.", 
            "The bulk mass drifts outward.", 
            "Center space remains effectively empty.", 
            "Robot scouts confirm the hollow interior."
        ])
        
        # Define elements
        outer_sphere = Circle(radius=1.5, color=WHITE, fill_opacity=0.1)
        shell = Circle(radius=1.45, color="#FF00FF", fill_opacity=0.3)
        inner_volume = Circle(radius=0.5, color="#FFA500", fill_opacity=0.5)
        
        # Assets
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        onion = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/onion.svg")
        
        # === Animation for Lecture Line 1 ===
        self.place_in_area(outer_sphere, "B2", "E5", scale_factor=0.6)
        self.place_at_grid(robot, "B5", scale_factor=0.3)
        self.play(FadeIn(outer_sphere), FadeIn(robot), self.lecture[0].animate.set_color("#FF00FF"))

        # === Animation for Lecture Line 2 ===
        self.place_in_area(shell, "B2", "E5", scale_factor=0.6)
        self.play(FadeIn(shell), self.lecture[1].animate.set_color("#FF00FF"))

        # === Animation for Lecture Line 3 ===
        self.place_in_area(inner_volume, "B2", "E5", scale_factor=0.6)
        self.play(Create(inner_volume), self.lecture[2].animate.set_color("#FFA500"))

        # === Animation for Lecture Line 4 ===
        density_curve = FunctionGraph(lambda x: 0.5 * np.exp(-10 * (x - 1.2)**2), x_range=[-2, 2], color=WHITE)
        self.place_at_grid(density_curve, "F3", scale_factor=0.4)
        self.play(Write(density_curve), self.lecture[3].animate.set_color(WHITE))

        # === Animation for Lecture Line 5 ===
        outward_arrows = VGroup(*[Arrow(start=ORIGIN, end=RIGHT*0.5, color="#ADFF2F").rotate(i * 2*PI/8, about_point=ORIGIN) for i in range(8)])
        self.place_in_area(outward_arrows, "A1", "F6", scale_factor=0.4)
        self.place_at_grid(onion, "A5", scale_factor=0.4)
        self.play(Create(outward_arrows), FadeIn(onion), self.lecture[4].animate.set_color("#ADFF2F"))
        self.wait(2)
