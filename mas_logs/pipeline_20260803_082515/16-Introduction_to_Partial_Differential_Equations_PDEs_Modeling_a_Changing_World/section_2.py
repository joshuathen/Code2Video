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
        # Setup layout with title and lecture lines
        title_text = "Prerequisite: The Concept of Multivariable Change"
        lecture_lines = [
            "Functions change with space and time.",
            "Partial derivatives measure change per variable.",
            "We freeze one variable to study another."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Color definitions
        surface_color = "#B0B0B0"
        x_deriv_color = "#55C1FF"
        t_deriv_color = "#FFD700"
        plane_color = WHITE
        
        # Load Assets
        surface_icon_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/surface.svg"
        plane_icon_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/plane.svg"
        
        surface_icon = SVGMobject(surface_icon_path).set_color(surface_color)
        self.place_at_grid(surface_icon, "A6", scale_factor=0.5)
        
        plane_icon = SVGMobject(plane_icon_path).set_color(x_deriv_color)
        self.place_at_grid(plane_icon, "B1", scale_factor=0.5)

        # Create 3D visualization objects
        axes = ThreeDAxes(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            z_range=[-1.5, 1.5, 1],
            x_length=4.5,
            y_length=4.5,
            z_length=3.5,
            axis_config={"include_tip": False, "stroke_width": 2}
        )
        
        # Surface function: f(x, t) = 0.8 * sin(x) * cos(t)
        def func(u, v):
            return axes.c2p(u, v, 0.8 * np.sin(u) * np.cos(v))

        surface = Surface(
            func,
            u_range=[-2, 2],
            v_range=[-2, 2],
            resolution=(12, 12),
            fill_color=surface_color,
            fill_opacity=0.7,
            checkerboard_colors=[surface_color, surface_color],
            stroke_width=0.5,
            stroke_color=WHITE
        )
        
        # 1. Elements for fixed time t (partial x)
        plane_t = Polygon(
            axes.c2p(-2, 0, -1.5), axes.c2p(2, 0, -1.5), axes.c2p(2, 0, 1.5), axes.c2p(-2, 0, 1.5),
            fill_color=plane_color, fill_opacity=0.2, stroke_width=1, stroke_color=WHITE
        )
        curve_t = ParametricFunction(
            lambda u: axes.c2p(u, 0, 0.8 * np.sin(u)),
            t_range=[-2, 2], color=x_deriv_color, stroke_width=4
        )
        tangent_t = Line(
            axes.c2p(-1, 0, -0.8), axes.c2p(1, 0, 0.8),
            color=x_deriv_color, stroke_width=6
        )
        v_elements_t = VGroup(plane_t, curve_t, tangent_t)

        # 2. Elements for fixed position x (partial t)
        plane_x = Polygon(
            axes.c2p(0, -2, -1.5), axes.c2p(0, 2, -1.5), axes.c2p(0, 2, 1.5), axes.c2p(0, -2, 1.5),
            fill_color=plane_color, fill_opacity=0.2, stroke_width=1, stroke_color=WHITE
        )
        curve_x = ParametricFunction(
            lambda v: axes.c2p(0, v, 0.8 * np.cos(v)),
            t_range=[-2, 2], color=t_deriv_color, stroke_width=4
        )
        v_elements_x = VGroup(plane_x, curve_x)

        # Master group for 3D alignment
        plot_group = VGroup(axes, surface, v_elements_t, v_elements_x)
        plot_group.rotate(70 * DEGREES, axis=RIGHT)
        plot_group.rotate(-30 * DEGREES, axis=OUT)
        
        # Apply positioning constraints (Issue 24: A1 to F6, scale 0.9)
        self.place_in_area(plot_group, "A1", "F6", scale_factor=0.9)
        
        # === Animation for Lecture Line 1 ===
        # "Functions change with space and time."
        self.play(
            self.lecture[0].animate.set_color(surface_color),
            FadeIn(axes),
            FadeIn(surface),
            FadeIn(surface_icon)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Partial derivatives measure change per variable."
        self.play(
            self.lecture[1].animate.set_color(x_deriv_color),
            FadeIn(plane_icon)
        )
        # Show intersection at fixed time t
        self.play(FadeIn(plane_t), Create(curve_t))
        self.play(Create(tangent_t))
        self.wait(2)
        self.play(FadeOut(v_elements_t), FadeOut(plane_icon))

        # === Animation for Lecture Line 3 ===
        # "We freeze one variable to study another."
        # Update plane_icon for the second plane (fixed x)
        plane_icon.set_color(t_deriv_color)
        
        self.play(
            self.lecture[2].animate.set_color(t_deriv_color),
            FadeIn(plane_icon),
            surface_icon.animate.set_color(t_deriv_color),
            surface.animate.set_color(t_deriv_color)
        )
        # Show intersection at fixed position x
        self.play(FadeIn(plane_x), Create(curve_x))
        
        # Animate the surface height changing (Asset pulse and geometry scale)
        self.play(
            plot_group.animate.scale(1.15, about_point=plot_group.get_center()),
            surface_icon.animate.scale(1.2),
            run_time=1.2,
            rate_func=there_and_back
        )
        self.wait(2)

        # Cleanup for scene transition
        self.play(
            FadeOut(plot_group),
            FadeOut(surface_icon),
            FadeOut(plane_icon),
            *[self.lecture[i].animate.set_color(WHITE) for i in range(3)]
        )
        self.wait(1)
