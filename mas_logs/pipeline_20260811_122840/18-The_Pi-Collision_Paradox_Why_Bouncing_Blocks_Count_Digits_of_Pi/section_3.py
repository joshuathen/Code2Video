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
        self.setup_layout("The Geometry of Collisions: The Phase Space Arc", 
                          ["Energy forms an ellipse in phase space.", 
                           "Collisions are boundary reflections in a slice.", 
                           "Trajectory arcs define the bounce count."])
        
        # Load Assets
        billiard_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/billiard.svg")
        table_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/table.svg")
        
        # Create visual elements
        axes = Axes(x_range=[-2, 2], y_range=[-2, 2], axis_config={"include_tip": False})
        self.place_in_area(axes, 'C2', 'F5', scale_factor=0.5)
        
        ellipse = Ellipse(width=3, height=1.5, color="#FF00FF")
        self.place_in_area(ellipse, 'C2', 'F5', scale_factor=0.5)
        
        line1 = Line(start=np.array([-1, -1, 0]), end=np.array([1, 1, 0]), color="#FFFF00").scale(0.5)
        self.place_at_grid(line1, "B4")
        
        arc = Arc(radius=1.5, start_angle=PI/4, angle=PI/2, color="#00FFFF")
        self.place_in_area(arc, 'C2', 'F5', scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.place_in_area(billiard_icon, 'A4', 'B5', scale_factor=0.4)
        self.play(Create(axes), Create(ellipse), FadeIn(billiard_icon))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF00FF"))
        self.play(Create(line1))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        self.play(Create(arc))
        
        # Fade out auxiliary, focus on final shape
        self.play(FadeOut(axes), FadeOut(line1), FadeOut(billiard_icon))
        self.place_in_area(table_icon, 'C2', 'F5', scale_factor=0.5)
        self.play(table_icon.animate.set_color("#00FF00"), FadeIn(table_icon))
        
        self.wait(1)
