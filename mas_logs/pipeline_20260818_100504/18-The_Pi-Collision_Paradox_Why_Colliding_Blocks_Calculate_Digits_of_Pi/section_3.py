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
        lecture_lines = [
            "Unfolding converts 1D collisions to 2D paths.",
            "Velocity vectors form a 2D particle plane.",
            "Reflections hit a potential barrier wedge."
        ]
        self.setup_layout("Mapping Physics to Geometry", lecture_lines)
        
        # Elements
        axes = Axes(
            x_range=[0, 5, 1], y_range=[0, 5, 1], 
            axis_config={"include_tip": True, "include_numbers": False},
            x_length=3.5, y_length=3.5
        )
        x_label = axes.get_x_axis_label(MathTex("v_1"))
        y_label = axes.get_y_axis_label(MathTex("v_2"))
        plane = VGroup(axes, x_label, y_label)
        
        barrier_wedge = axes.plot(lambda x: 5 - x, color=YELLOW, x_range=[0, 5])
        
        # Asset usage: /scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg
        try:
            particle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg")
        except:
            particle = Dot(color=RED, radius=0.1)

        path = VMobject(color=BLUE)
        path.set_points_as_corners([axes.c2p(0.5, 0.5), axes.c2p(2.5, 2.5)])

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]))
        # Fix overlapping per VideoCritic
        self.place_in_area(plane, 'B4', 'F6', scale_factor=0.6)
        self.play(Create(plane))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(GRAY), FadeIn(self.lecture[1]))
        # Fix axes scaling per VideoCritic
        axes_group = VGroup(axes, x_label, y_label)
        self.place_at_grid(axes_group, 'C4', scale_factor=0.7)
        self.play(Create(axes_group))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(GRAY), FadeIn(self.lecture[2]))
        # Fix wedge encroaching per VideoCritic
        self.place_at_grid(barrier_wedge, 'D3', scale_factor=0.8)
        self.play(Create(barrier_wedge))
        
        self.play(
            FadeIn(particle),
            MoveAlongPath(particle, path),
            Create(path),
            run_time=2
        )
        self.wait(1)
