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
        self.setup_layout("Prerequisite: The Average Rate of Change", [
            "Average speed is distance divided by time.", 
            "Visualize the slope of a secant line.", 
            "Cheetah sprints 100 meters in 5 seconds."
        ])
        
        # Assets
        cheetah = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cheetah.svg")
        
        # Create Axes
        axes = Axes(x_range=[0, 6, 1], y_range=[0, 120, 20], axis_config={"color": "#FFFFFF"}).scale(0.4)
        group = VGroup(axes)
        
        # === Animation for Lecture Line 1 ===
        # Color the first line
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        
        # Fix: Using B1-E4 area as per Issue 17
        self.place_in_area(group, 'B1', 'E4', scale_factor=0.5)
        self.add(group)
        
        # Place cheetah
        self.place_at_grid(cheetah, 'F6', scale_factor=0.3)
        self.play(FadeIn(cheetah))

        # === Animation for Lecture Line 2 ===
        # Color the second line
        self.play(self.lecture[1].animate.set_color("#FF00FF"))
        
        curve = axes.plot(lambda x: 4 * x**2, x_range=[0, 5], color="#FFFF00")
        p = Dot(axes.c2p(1, 4), color="#00FFFF")
        q = Dot(axes.c2p(5, 100), color="#00FFFF")
        secant = Line(axes.c2p(1, 4), axes.c2p(5, 100), color="#FF00FF")
        
        self.play(Create(curve))
        self.play(FadeIn(p), FadeIn(q))
        self.play(Create(secant))
        
        # === Animation for Lecture Line 3 ===
        # Color the third line
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        
        # Use MathTex for v=d/t as mentioned in storyboard
        formula = MathTex(r"v = \frac{d}{t}", color="#FFFF00")
        self.place_at_grid(formula, 'A2', scale_factor=0.8)
        self.play(Write(formula))
        
        # Slope formula
        slope_label = MathTex(r"m = \frac{f(x+h)-f(x)}{h}", color="#FFFFFF")
        # Fix: F2 position as per Issue 19
        self.place_at_grid(slope_label, 'F2', scale_factor=0.7)
        self.play(Write(slope_label))
        
        self.wait(2)
