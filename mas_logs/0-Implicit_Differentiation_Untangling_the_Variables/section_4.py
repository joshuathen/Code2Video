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
        # 1. Setup layout
        title_text = "Guided Practice: The Laser Tangent"
        lecture_lines = [
            "At point (3, 4), our derivative is negative three-fourths.",
            "This value represents the exact slope of the tangent.",
            "See the laser beam tilt perfectly against the circle."
        ]
        self.setup_layout(title_text, lecture_lines)

        # 2. Prepare graph elements (grouped for grid positioning)
        # Using Axes with a range that accommodates a radius-5 circle and its tangent.
        axes = Axes(
            x_range=[-10, 10, 2],
            y_range=[-10, 10, 2],
            x_length=5,
            y_length=5,
            axis_config={"color": WHITE},
            tips=False
        )
        
        # Calculate Manim-space radius for the circle x^2 + y^2 = 25
        origin_pos = axes.coords_to_point(0, 0)
        radius_pos = axes.coords_to_point(5, 0)
        radius_val = np.linalg.norm(radius_pos - origin_pos)
        
        circle = Circle(radius=radius_val, color="#00FFFF")
        circle.move_to(origin_pos)
        
        dot = Dot(point=axes.coords_to_point(3, 4), color="#FF0000")
        label_p = Text("(3, 4)", font_size=16, color="#FF0000")
        label_p.next_to(dot, UR, buff=0.1)
        
        # Tangent line (laser beam): y - 4 = -0.75(x - 3) => y = -0.75x + 6.25
        # Define two points on the line within the axes range.
        # x=-1 => y=7; x=7 => y=1
        p_start = axes.coords_to_point(-1, 7)
        p_end = axes.coords_to_point(7, 1)
        laser_beam = Line(p_start, p_end, color="#FFFF00", stroke_width=4)
        
        # Bundle graph components into a VGroup for precise positioning within the grid
        graph_group = VGroup(axes, circle, dot, label_p, laser_beam)
        # Place graph in the left column-set of the visual grid (A1 to F3)
        self.place_in_area(graph_group, "A1", "F3", scale_factor=0.8)

        # 3. Prepare math elements (will be displayed in the right column-set)
        # Using Text instead of MathTex to avoid potential LaTeX environment issues.
        
        # Substitution step
        sub_tex = Text("2(3) + 2(4)(dy/dx) = 0", font_size=22, color=WHITE)
        self.place_at_grid(sub_tex, "B5", scale_factor=1.0)

        # Simplified step
        calc_tex = Text("8(dy/dx) = -6", font_size=22, color=WHITE)
        self.place_at_grid(calc_tex, "C5", scale_factor=1.0)

        # Final slope value
        final_tex = Text("dy/dx = -3/4", font_size=24, color="#FFFF00")
        self.place_at_grid(final_tex, "D5", scale_factor=1.1)

        # 4. Animations

        # === Animation for Lecture Line 1 ===
        # Highlight lecture line and draw the visual environment + calculation
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        self.play(Create(axes), Create(circle))
        self.play(FadeIn(dot), Write(label_p))
        self.wait(0.5)
        
        # Display the math steps sequentially
        self.play(Write(sub_tex))
        self.play(Write(calc_tex))
        self.play(Write(final_tex))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Emphasize the connection between the derivative value and the slope concept
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        self.play(Indicate(final_tex, color="#FFFF00"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Final visual confirmation with the laser beam (tangent line)
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        self.play(Create(laser_beam))
        self.wait(2)
