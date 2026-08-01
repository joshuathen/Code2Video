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
        # Initial Setup
        title = "The Winding Number: The Mathematical Scout"
        lines = [
            'We trace a closed loop in the input domain.', 
            'The function maps this to a path in output.', 
            'A vector tracks how many times we circle zero.', 
            'A non-zero count guarantees a root inside the loop.', 
            'Zero winding means no roots exist within the boundary.'
        ]
        self.setup_layout(title, lines)

        # --- Colors ---
        DOMAIN_COLOR = YELLOW
        CODOMAIN_COLOR = TEAL
        VECTOR_COLOR = ORANGE
        ROOT_COLOR = GREEN
        NO_ROOT_COLOR = RED

        # --- Assets Construction ---
        # Assets Path
        scout_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/scout.svg"

        # Domain Plane
        domain_axes = Axes(
            x_range=[-2, 2, 1], y_range=[-2, 2, 1],
            x_length=2.8, y_length=2.8,
            axis_config={"color": GREY_D, "include_tip": False}
        )
        domain_label = Text("Domain (z)", font_size=18, color=DOMAIN_COLOR)
        domain_group = VGroup(domain_axes, domain_label).arrange(UP, buff=0.2)
        self.place_in_area(domain_group, "B1", "D3", scale_factor=0.8)

        # Codomain Plane
        codomain_axes = Axes(
            x_range=[-2, 2, 1], y_range=[-2, 2, 1],
            x_length=2.8, y_length=2.8,
            axis_config={"color": GREY_D, "include_tip": False}
        )
        codomain_label = Text("Codomain (f(z))", font_size=18, color=CODOMAIN_COLOR)
        codomain_group = VGroup(codomain_axes, codomain_label).arrange(UP, buff=0.2)
        self.place_in_area(codomain_group, "B4", "D6", scale_factor=0.8)

        # Elements for the paths
        root_pos = np.array([0.5, 0.5, 0])
        input_loop = Circle(radius=0.8, color=DOMAIN_COLOR).move_to(domain_axes.c2p(*root_pos))
        output_loop = Circle(radius=0.8, color=CODOMAIN_COLOR).move_to(codomain_axes.c2p(0, 0, 0))
        
        # Scout Mobjects
        scout_domain = SVGMobject(scout_path, height=0.3, color=DOMAIN_COLOR)
        scout_codomain = SVGMobject(scout_path, height=0.3, color=CODOMAIN_COLOR)

        # Tracker for motion
        t_tracker = ValueTracker(0)

        # --- Animation for Lecture Line 1 ---
        self.play(self.lecture[0].animate.set_color(DOMAIN_COLOR))
        
        scout_domain.move_to(input_loop.point_from_proportion(0))
        self.play(Create(domain_axes), FadeIn(domain_label))
        self.play(Create(input_loop))
        self.play(FadeIn(scout_domain))
        
        scout_domain.add_updater(lambda m: m.move_to(input_loop.point_from_proportion(t_tracker.get_value() % 1.0)))
        self.play(t_tracker.animate.set_value(0.5), run_time=1.5, rate_func=linear)
        self.wait(0.5)

        # --- Animation for Lecture Line 2 ---
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(CODOMAIN_COLOR)
        )

        scout_codomain.move_to(output_loop.point_from_proportion(t_tracker.get_value() % 1.0))
        self.play(Create(codomain_axes), FadeIn(codomain_label))
        self.play(Create(output_loop))
        self.play(FadeIn(scout_codomain))
        
        scout_codomain.add_updater(lambda m: m.move_to(output_loop.point_from_proportion(t_tracker.get_value() % 1.0)))
        self.play(t_tracker.animate.set_value(1.0), run_time=1.5, rate_func=linear)
        self.wait(0.5)

        # --- Animation for Lecture Line 3 ---
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(VECTOR_COLOR)
        )

        # Origin point in codomain
        origin_codomain = codomain_axes.c2p(0, 0, 0)
        
        leash_vector = always_redraw(lambda: Arrow(
            start=origin_codomain,
            end=scout_codomain.get_center(),
            buff=0,
            color=VECTOR_COLOR,
            stroke_width=4,
            max_tip_length_to_length_ratio=0.15
        ))

        self.play(GrowArrow(leash_vector))
        self.play(t_tracker.animate.set_value(2.0), run_time=3, rate_func=linear)
        self.wait(0.5)

        # --- Animation for Lecture Line 4 ---
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(ROOT_COLOR)
        )

        root_dot = Dot(domain_axes.c2p(*root_pos), color=ROOT_COLOR, radius=0.08)
        root_text = Text("Root", font_size=16, color=ROOT_COLOR).next_to(root_dot, UR, buff=0.1)
        
        winding_text = Text("Winding Number = 1", font_size=24, color=ROOT_COLOR)
        self.place_in_area(winding_text, 'F2', 'F5', scale_factor=0.8)

        self.play(FadeIn(root_dot), FadeIn(root_text))
        self.play(Write(winding_text))
        self.play(Indicate(root_dot), Indicate(winding_text))
        self.wait(1)

        # --- Animation for Lecture Line 5 ---
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(NO_ROOT_COLOR)
        )

        # Define non-root case
        new_loop_center = np.array([-1.2, -1.2, 0])
        new_input_loop = Circle(radius=0.5, color=DOMAIN_COLOR).move_to(domain_axes.c2p(*new_loop_center))
        # Image loop far from origin
        new_output_center = np.array([1.5, 1.5, 0])
        new_output_loop = Circle(radius=0.4, color=CODOMAIN_COLOR).move_to(codomain_axes.c2p(*new_output_center))
        
        new_winding_text = Text("Winding Number = 0", font_size=24, color=NO_ROOT_COLOR)
        self.place_in_area(new_winding_text, 'F2', 'F5', scale_factor=0.8)

        # Stop updaters for transform
        scout_domain.clear_updaters()
        scout_codomain.clear_updaters()

        self.play(
            Transform(input_loop, new_input_loop),
            Transform(output_loop, new_output_loop),
            Transform(winding_text, new_winding_text),
            FadeOut(root_dot),
            FadeOut(root_text),
            scout_domain.animate.move_to(new_input_loop.point_from_proportion(0)),
            scout_codomain.animate.move_to(new_output_loop.point_from_proportion(0)),
            run_time=2
        )
        
        t_tracker.set_value(0)
        scout_domain.add_updater(lambda m: m.move_to(input_loop.point_from_proportion(t_tracker.get_value() % 1.0)))
        scout_codomain.add_updater(lambda m: m.move_to(output_loop.point_from_proportion(t_tracker.get_value() % 1.0)))
        
        self.play(t_tracker.animate.set_value(1.0), run_time=3, rate_func=linear)
        self.wait(2)

        # Cleanup
        self.play(
            FadeOut(domain_group), FadeOut(codomain_group),
            FadeOut(input_loop), FadeOut(output_loop),
            FadeOut(scout_domain), FadeOut(scout_codomain),
            FadeOut(leash_vector), FadeOut(winding_text)
        )
