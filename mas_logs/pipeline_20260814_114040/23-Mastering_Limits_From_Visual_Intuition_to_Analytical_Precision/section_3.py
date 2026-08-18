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
        self.setup_layout("Computational Shortcut: L'Hôpital's Rule", 
                          ["L'Hôpital's rule solves indeterminate limit forms.", 
                           "Functions must be differentiable at the point.", 
                           "Compare slopes to find the limit."])
        
        # Elements
        axes = Axes(x_range=[-2, 2], y_range=[-2, 2], axis_config={"include_tip": True}).scale(0.5)
        f_sin = axes.plot(lambda x: np.sin(x), color=BLUE)
        f_id = axes.plot(lambda x: x, color=YELLOW)
        group = VGroup(axes, f_sin, f_id)
        
        pencil = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pencil.svg").set_color(ORANGE)
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg").set_color("#DDA0DD")
        
        indeterminate_form = MathTex(r"0/0").set_color(ORANGE)
        slope_text = MathTex(r"\frac{f'(0)}{g'(0)} = 1").scale(0.7)

        # Positioning
        self.place_in_area(group, "B4", "F6", scale_factor=0.6)
        self.place_at_grid(pencil, "A4", scale_factor=0.5)
        self.place_at_grid(ruler, "A5", scale_factor=0.5)
        self.place_at_grid(indeterminate_form, "A6", scale_factor=0.8)
        self.place_at_grid(slope_text, "E1", scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(ORANGE))
        self.play(FadeIn(pencil), Write(indeterminate_form))
        self.play(Create(group))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#DDA0DD"))
        self.play(FadeIn(ruler))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.play(FadeIn(slope_text))
        self.wait(2)
