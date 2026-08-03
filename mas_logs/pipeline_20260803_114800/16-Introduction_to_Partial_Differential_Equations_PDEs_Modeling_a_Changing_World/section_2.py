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
        self.setup_layout("Prerequisite Bridge: From Derivatives to Partials", [
            "Imagine hiking across a steep, 3D mountain range.",
            "A partial derivative finds the slope in one direction.",
            "We hold all other variables constant while measuring change."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Highlight lecture line
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        
        # Draw a 2D curve with a tangent line #00FF00
        axes = Axes(
            x_range=[-2, 2], y_range=[0, 4], 
            axis_config={"include_tip": False},
            x_length=3, y_length=3
        ).set_color(GRAY)
        curve = axes.plot(lambda x: 0.5 * x**2 + 1, color="#00FF00")
        
        # Tangent line at x=1
        point_coord = 1
        point = axes.coords_to_point(point_coord, 0.5 * point_coord**2 + 1)
        dot = Dot(point, color=WHITE)
        tangent = TangentLine(curve, alpha=0.75, length=3, color="#00FF00")
        
        graph_group = VGroup(axes, curve, dot, tangent)
        self.place_in_area(graph_group, "B2", "E5", scale_factor=0.8)
        
        self.play(Create(axes), Create(curve))
        self.play(FadeIn(dot), Create(tangent))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Highlight lecture line
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#00BFFF")
        )
        
        # Show a grid where only one axis is highlighted #00BFFF
        new_grid = NumberPlane(
            x_range=[-2, 2], y_range=[-2, 2],
            x_length=4, y_length=4,
            background_line_style={"stroke_opacity": 0.2}
        )
        self.place_in_area(new_grid, "B2", "E5", scale_factor=0.8)
        
        # Highlight one axis
        highlight_axis = Line(
            new_grid.coords_to_point(-2, 0),
            new_grid.coords_to_point(2, 0),
            color="#00BFFF", stroke_width=6
        )
        
        self.play(
            FadeOut(graph_group),
            Create(new_grid)
        )
        self.play(Create(highlight_axis))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Highlight lecture line
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FF69B4")
        )
        
        # Display "∂f/∂x" and "∂f/∂y" with arrows pointing along axes
        dfdx = MathTex(r"\frac{\partial f}{\partial x}", color="#FF69B4")
        dfdy = MathTex(r"\frac{\partial f}{\partial y}", color="#FF69B4")
        
        # Fix for Issue 23: Move dfdx to A3
        self.place_at_grid(dfdx, "A3", scale_factor=1.2)
        # Fix for Issue 24: Move dfdy to D2
        self.place_at_grid(dfdy, "D2", scale_factor=1.2)
        
        arrow_x = Arrow(
            new_grid.coords_to_point(0, 0),
            new_grid.coords_to_point(1.5, 0),
            color="#FF69B4", buff=0
        )
        arrow_y = Arrow(
            new_grid.coords_to_point(0, 0),
            new_grid.coords_to_point(0, 1.5),
            color="#FF69B4", buff=0
        )
        
        self.play(
            Write(dfdx),
            Write(dfdy),
            GrowArrow(arrow_x),
            GrowArrow(arrow_y)
        )
        self.wait(3)
