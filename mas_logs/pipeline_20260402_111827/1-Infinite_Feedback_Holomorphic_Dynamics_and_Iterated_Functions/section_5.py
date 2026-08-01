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
        # Setup layout
        lecture_lines_text = [
            "The Mandelbrot set maps every possible value of c.",
            "It acts as a master catalog for all Julia sets.",
            "Points inside the set produce connected, intricate Julia shapes.",
            "Outside points create disconnected clouds of mathematical dust.",
            "It represents the bridge between order and chaotic feedback."
        ]
        self.setup_layout("The Mandelbrot Set: The Master Map", lecture_lines_text)

        # Colors
        COLOR_AXES = "#888888"
        COLOR_MANDELBROT = "#AA00FF"
        COLOR_INSIDE = "#00FFFF"
        COLOR_OUTSIDE = "#FF8C00"
        COLOR_ZOOM = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        # Display the complex parameter plane with 'c' axes in #888888.
        self.lecture[0].set_color(COLOR_AXES)
        
        axes = Axes(
            x_range=[-2.5, 1.5, 1],
            y_range=[-1.5, 1.5, 1],
            x_length=4.5,
            y_length=4.5,
            axis_config={"color": COLOR_AXES, "include_tip": True}
        )
        labels = axes.get_axis_labels(
            x_label=Text("Re(c)", font_size=18), 
            y_label=Text("Im(c)", font_size=18)
        )
        plane_group = VGroup(axes, labels)
        
        # Fixed placement as per Issue 38 to avoid title overlap
        self.place_in_area(plane_group, "B2", "F6", scale_factor=0.8)
        
        self.play(Create(axes), Write(labels))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Fill the Mandelbrot set region where orbits stay bounded in #AA00FF.
        self.lecture[1].set_color(COLOR_MANDELBROT)
        
        # Stylized Mandelbrot Set construction using geometric primitives
        cardioid_points = [
            axes.c2p(
                0.5 * np.cos(t) - 0.25 * np.cos(2*t) + 0.12, 
                0.5 * np.sin(t) - 0.25 * np.sin(2*t)
            )
            for t in np.linspace(0, TAU, 100)
        ]
        main_cardioid = Polygon(*cardioid_points, fill_opacity=0.8, fill_color=COLOR_MANDELBROT, stroke_width=1, stroke_color=COLOR_MANDELBROT)
        
        radius_unit = (axes.x_axis.get_unit_size())
        main_bulb = Circle(radius=0.25 * radius_unit, color=COLOR_MANDELBROT, fill_opacity=0.8)
        main_bulb.move_to(axes.c2p(-1, 0))
        
        bulb_top = Circle(radius=0.1 * radius_unit, color=COLOR_MANDELBROT, fill_opacity=0.8).move_to(axes.c2p(-0.12, 0.75))
        bulb_bottom = bulb_top.copy().move_to(axes.c2p(-0.12, -0.75))
        
        mandelbrot_shape = VGroup(main_cardioid, main_bulb, bulb_top, bulb_bottom)
        
        self.play(FadeIn(mandelbrot_shape))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight an interior 'c' point and its connected Julia set thumbnail.
        self.lecture[2].set_color(COLOR_INSIDE)
        
        c_in_coord = [-0.1, 0.2]
        c_inside_dot = Dot(axes.c2p(*c_in_coord), color=COLOR_INSIDE)
        c_in_txt = Text("c inside", font_size=16, color=COLOR_INSIDE)
        c_in_txt.next_to(c_inside_dot, UP, buff=0.1)
        
        # Connected Julia Thumbnail (stylized circle representation)
        julia_conn = Circle(radius=0.4, color=COLOR_INSIDE, fill_opacity=0.3)
        julia_conn_txt = Text("Connected Julia", font_size=14, color=COLOR_INSIDE).next_to(julia_conn, DOWN, buff=0.1)
        julia_thumb_1 = VGroup(julia_conn, julia_conn_txt)
        self.place_at_grid(julia_thumb_1, "B6", scale_factor=0.7)
        
        self.play(FadeIn(c_inside_dot), Write(c_in_txt))
        self.play(FadeIn(julia_thumb_1))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Highlight an exterior 'c' point and its disconnected dust Julia set.
        self.lecture[3].set_color(COLOR_OUTSIDE)
        
        c_out_coord = [0.6, 0.6]
        c_outside_dot = Dot(axes.c2p(*c_out_coord), color=COLOR_OUTSIDE)
        c_out_txt = Text("c outside", font_size=16, color=COLOR_OUTSIDE)
        c_out_txt.next_to(c_outside_dot, RIGHT, buff=0.1)
        
        # Dust Julia Thumbnail (stylized point cloud)
        dots = [Dot(radius=0.015, color=COLOR_OUTSIDE).move_to([np.random.uniform(-0.3, 0.3), np.random.uniform(-0.3, 0.3), 0]) for _ in range(30)]
        julia_dust = VGroup(*dots)
        julia_dust_txt = Text("Julia Dust", font_size=14, color=COLOR_OUTSIDE).next_to(julia_dust, DOWN, buff=0.1)
        julia_thumb_2 = VGroup(julia_dust, julia_dust_txt)
        self.place_at_grid(julia_thumb_2, "D6", scale_factor=0.7)

        self.play(FadeIn(c_outside_dot), Write(c_out_txt))
        self.play(FadeIn(julia_thumb_2))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Zoom into the Mandelbrot boundary to reveal infinite, repeating detail.
        self.lecture[4].set_color(COLOR_ZOOM)
        
        # Zoom target near the main cardioid boundary
        zoom_target = axes.c2p(-0.75, 0.1)
        zoom_group = VGroup(plane_group, mandelbrot_shape, c_inside_dot, c_outside_dot)
        
        # Clean up thumbnails for focus
        self.play(
            FadeOut(julia_thumb_1), 
            FadeOut(julia_thumb_2), 
            FadeOut(c_in_txt), 
            FadeOut(c_out_txt)
        )
        
        # Zoom animation
        self.play(
            zoom_group.animate.scale(5, about_point=zoom_target).shift(self.grid["D4"] - zoom_target),
            run_time=3
        )
        
        # Represent repeating infinite detail with nested circles
        mini_bulbs = VGroup(*[
            Circle(radius=0.04 * i, color=COLOR_ZOOM, fill_opacity=0.6).move_to(self.grid["D4"] + RIGHT*0.3*i)
            for i in range(1, 4)
        ])
        self.play(FadeIn(mini_bulbs))
        
        self.wait(2)
