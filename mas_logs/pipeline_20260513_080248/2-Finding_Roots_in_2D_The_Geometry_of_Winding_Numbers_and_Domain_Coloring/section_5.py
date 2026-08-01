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
        # Initial Setup with correct script lines
        title = "Real-World Application: Stability and Design"
        lines = [
            "Nyquist plots use winding numbers to test system stability.",
            "If the origin is encircled, the system is unstable.",
            "These topological tools solve complex problems across many fields."
        ]
        self.setup_layout(title, lines)

        # Colors for highlights
        color1 = YELLOW_A
        color2 = BLUE_A
        color3 = LIGHT_PINK

        # === Animation for Lecture Line 1 ===
        # Display a complex plane with a Nyquist plot (a closed loop) and a critical point (-1,0).
        self.lecture[0].set_color(color1)
        
        plane = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=4,
            y_length=4,
            background_line_style={"stroke_opacity": 0.4},
            axis_config={"include_tip": True}
        )
        self.place_in_area(plane, 'B2', 'E5')
        
        # Nyquist plot (a loop) encircling -1
        nyquist_loop = ParametricFunction(
            lambda t: plane.coords_to_point(
                1.5 * np.cos(t) - 0.5 * np.cos(2*t),
                1.5 * np.sin(t) - 0.5 * np.sin(2*t),
                0
            ),
            t_range=[0, TAU],
            color=color1
        )
        
        # Critical point at (-1, 0)
        crit_point_coord = plane.coords_to_point(-1, 0, 0)
        crit_dot = Dot(crit_point_coord, color=RED)
        crit_label = Text("(-1, 0)", font_size=16, color=RED)
        # Position label relative to dot using a grid nearby or logic
        crit_label.next_to(crit_dot, DOWN, buff=0.1)
        
        plot_label = Text("Nyquist Plot", font_size=20, color=color1)
        # Issue 37: Area positioning for plot label
        self.place_in_area(plot_label, 'A3', 'A5', scale_factor=0.8)

        self.play(FadeIn(plane), Create(nyquist_loop), FadeIn(crit_dot), FadeIn(crit_label), FadeIn(plot_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Animate the winding number around (-1,0) to determine if a system [Asset] is stable.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(color2)
        
        # Load Asset (Issue 25)
        system_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/system.svg").set_color(color2)
        self.place_at_grid(system_icon, 'F2', scale_factor=0.5)

        # Winding vector from (-1,0) to moving point on loop
        tracer_point = Dot(nyquist_loop.get_start(), radius=0.05, color=color2)
        winding_vector = Line(crit_point_coord, tracer_point.get_center(), color=color2, stroke_width=2)
        
        # Winding number counter (Issue 35)
        w_text = Text("Winding Number: ", font_size=20, color=color2)
        w_val = Text("1", font_size=20, color=color2).next_to(w_text, RIGHT)
        w_group = VGroup(w_text, w_val)
        self.place_in_area(w_group, 'F3', 'F5', scale_factor=0.8)

        self.play(FadeIn(w_group), FadeIn(system_icon))
        
        # Custom update for vector - keeping it simple for MAS constraints
        def update_vector(mob):
            mob.put_start_and_end_on(crit_point_coord, tracer_point.get_center())

        winding_vector.add_updater(update_vector)
        self.add(winding_vector)
        
        self.play(MoveAlongPath(tracer_point, nyquist_loop), run_time=4, rate_func=linear)
        self.wait(1)
        
        winding_vector.remove_updater(update_vector)
        self.play(
            FadeOut(tracer_point), FadeOut(winding_vector), FadeOut(w_group), 
            FadeOut(plane), FadeOut(nyquist_loop), FadeOut(crit_dot), 
            FadeOut(crit_label), FadeOut(plot_label), FadeOut(system_icon)
        )

        # === Animation for Lecture Line 3 ===
        # Fade to a final summary graphic showing the domain coloring and recursive squares.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(color3)

        # Create a grid of colored squares to represent domain coloring
        colors = [RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE]
        domain_grid = VGroup()
        for i in range(4):
            for j in range(4):
                sq = Square(side_length=0.6, fill_opacity=0.8, stroke_width=1)
                sq.set_fill(colors[(i+j) % len(colors)])
                domain_grid.add(sq)
        domain_grid.arrange_in_grid(rows=4, cols=4, buff=0.05)
        self.place_in_area(domain_grid, 'B2', 'E5')

        # Create recursive subdivision
        sub_grid = VGroup()
        for i in range(2):
            for j in range(2):
                sq = Square(side_length=0.25, fill_opacity=1.0, stroke_width=0.5, color=WHITE)
                sq.set_fill(color3)
                sub_grid.add(sq)
        sub_grid.arrange_in_grid(rows=2, cols=2, buff=0.02)
        sub_grid.move_to(domain_grid[5].get_center())

        # Issue 36: Area positioning for graphics label
        graphics_label = Text("Recursive Grid Search", font_size=20, color=color3)
        self.place_in_area(graphics_label, 'A3', 'A5', scale_factor=0.8)

        self.play(FadeIn(domain_grid), FadeIn(graphics_label))
        self.play(domain_grid[5].animate.set_fill(opacity=0.2))
        self.play(Create(sub_grid))
        
        # Add final level of recursion
        tiny_grid = VGroup()
        for i in range(2):
            for j in range(2):
                sq = Square(side_length=0.1, fill_opacity=1.0, stroke_width=0.2, color=WHITE)
                sq.set_fill(YELLOW)
                tiny_grid.add(sq)
        tiny_grid.arrange_in_grid(rows=2, cols=2, buff=0.01)
        tiny_grid.move_to(sub_grid[0].get_center())
        
        self.play(sub_grid[0].animate.set_fill(opacity=0.2))
        self.play(Create(tiny_grid))
        
        self.wait(2)
        
        # Final cleanup
        self.lecture[2].set_color(WHITE)
        self.wait(1)
