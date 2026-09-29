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
        self.setup_layout("Visual Application: Tangent Lines", 
                          ["Consider the Folium of Descartes curve.", 
                           "Pick a point (3,3) on it.", 
                           "Use dy/dx to find tangent line."])
        
        # Folium of Descartes: x^3 + y^3 - 6xy = 0
        def folium_func(t):
            x = 6 * t / (1 + t**3)
            y = 6 * t**2 / (1 + t**3)
            return np.array([x, y, 0])

        # Axes
        axes = Axes(x_range=[-2, 6, 1], y_range=[-2, 6, 1], x_length=4, y_length=4)
        self.place_in_area(axes, 'B3', 'E4', scale_factor=0.9)

        # Asset
        folium_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/folium.svg")
        
        # Combined graphic for centering
        curve = ParametricFunction(folium_func, t_range=[-5, 5], color=BLUE)
        curve.scale(0.8).move_to(axes.c2p(1.5, 1.5))
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFD700")
        self.play(Create(curve), FadeIn(folium_icon.scale(0.5).to_corner(UR)))
        self.add(axes)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FFFF")
        point = Dot(axes.c2p(3, 3), color=RED)
        point_label = MathTex(r"(3,3)", font_size=24).next_to(point, UP)
        self.play(FadeIn(point), Write(point_label))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FF00")
        tangent = Line(start=axes.c2p(1, 5), end=axes.c2p(5, 1), color=YELLOW)
        slope_label = MathTex(r"\frac{dy}{dx} = -1", color="#00FF00", font_size=24)
        
        self.place_at_grid(slope_label, 'D5', scale_factor=0.8)
        
        # Grouping for balancing
        combined_graphic = VGroup(curve, axes, point, point_label, tangent)
        self.place_in_area(combined_graphic, 'A3', 'F5', scale_factor=0.75)
        
        self.play(Create(tangent), Write(slope_label))
        self.wait(2)
