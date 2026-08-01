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

class Section5Scene(TeachingScene):
    def construct(self):
        # Setup layout with teaching script lines
        self.setup_layout("The Grand Finale: Walking π Distance", [
            'Now, let us walk a distance of pi.', 
            'This represents a half-turn around the circle.', 
            'We start at one on the real axis.', 
            'Halfway around, we land exactly on negative one.', 
            'Therefore, e to the i pi equals negative one.'
        ])

        # Define mathematical components
        # Issue 35: Positioning Euler's formula in area A2-A5
        euler_formula = Text("e^iθ = cos(θ) + i sin(θ)", font_size=32)
        self.place_in_area(euler_formula, 'A2', 'A5', scale_factor=0.8)

        # Issue 36: Positioning complex plane group in area B2-E5
        axes = ComplexPlane(
            x_range=[-2, 2, 1],
            y_range=[-1.5, 1.5, 1],
            x_length=4.5,
            y_length=3.5,
            axis_config={"include_tip": True}
        )
        labels = axes.get_axis_labels(
            x_label=Text("Re", font_size=18), 
            y_label=Text("Im", font_size=18)
        )
        circle = Circle(radius=axes.x_axis.get_unit_size(), color=BLUE_B)
        circle.move_to(axes.get_origin())
        
        complex_plane_group = VGroup(axes, labels, circle)
        self.place_in_area(complex_plane_group, 'B2', 'E5', scale_factor=0.9)

        # Issue 37: Positioning identity result in area F2-F5
        identity_result = Text("e^iπ = -1", color="#FFD700", font_size=40)
        self.place_in_area(identity_result, 'F2', 'F5', scale_factor=0.8)

        # Dynamic elements relative to axes
        # Yellow arc of length π appearing above the circle
        arc = Arc(
            radius=axes.x_axis.get_unit_size(), 
            start_angle=0, 
            angle=PI, 
            color="#FFFF00",
            stroke_width=6
        )
        arc.move_to(axes.get_origin())
        
        # White point starting at (1, 0)
        dot = Dot(axes.n2p(1 + 0j), color=WHITE)

        # === Animation for Lecture Line 1 ===
        # 'Now, let us walk a distance of pi.'
        self.play(
            Write(euler_formula),
            Create(axes),
            Create(labels),
            Create(circle),
            self.lecture[0].animate.set_color("#FFFF00")
        )
        self.play(Create(arc), run_time=1.5)

        # === Animation for Lecture Line 2 ===
        # 'This represents a half-turn around the circle.'
        self.play(
            Indicate(arc, color="#FFFF00"),
            self.lecture[1].animate.set_color("#FFFF00")
        )
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # 'We start at one on the real axis.'
        self.play(
            FadeIn(dot),
            self.lecture[2].animate.set_color("#FFFFFF")
        )
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        # 'Halfway around, we land exactly on negative one.'
        self.play(
            MoveAlongPath(dot, arc),
            self.lecture[3].animate.set_color("#FFFFFF"),
            run_time=2.5
        )
        self.play(Indicate(dot, scale_factor=1.5))

        # === Animation for Lecture Line 5 ===
        # 'Therefore, e to the i pi equals negative one.'
        self.play(
            Write(identity_result),
            self.lecture[4].animate.set_color("#FFD700")
        )
        self.play(Indicate(identity_result, color="#FFD700"))
        self.wait(3)

        # Cleanup
        self.play(
            FadeOut(euler_formula),
            FadeOut(complex_plane_group),
            FadeOut(identity_result),
            FadeOut(arc),
            FadeOut(dot)
        )
