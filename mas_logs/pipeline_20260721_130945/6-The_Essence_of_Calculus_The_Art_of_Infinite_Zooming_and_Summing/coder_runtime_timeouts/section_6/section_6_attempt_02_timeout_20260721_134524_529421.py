from manim import *
import numpy as np

# Use the provided TeachingScene base class without modification.
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

class Section6Scene(TeachingScene):
    def construct(self):
        # Section data
        title_text = "Conclusion: Calculus in the Real World"
        lecture_lines = [
            "Calculus models everything from planetary orbits to medicine.",
            "It is the essential language of our moving universe.",
            "Master it to understand how everything truly changes."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        COLOR_LINE1 = WHITE
        COLOR_LINE2 = "#00FFFF" # Cyan
        COLOR_LINE3 = "#FFFF00" # Yellow
        COLOR_FIRE = "#FF4500"
        COLOR_GRAPH = "#FF0000"

        # === Animation for Lecture Line 1 ===
        # Highlight Line 1
        self.play(self.lecture[0].animate.set_color(COLOR_LINE1), run_time=0.5)

        # Rocket construction
        rocket_body = Rectangle(width=0.4, height=1.0, color=WHITE, fill_opacity=1)
        rocket_tip = Triangle(color=WHITE, fill_opacity=1).scale(0.2)
        rocket_tip.next_to(rocket_body, UP, buff=0)
        rocket_fin_l = Triangle(color=WHITE, fill_opacity=1).scale(0.15).rotate(-PI/2)
        rocket_fin_l.next_to(rocket_body, LEFT, buff=0, aligned_edge=DOWN)
        rocket_fin_r = Triangle(color=WHITE, fill_opacity=1).scale(0.15).rotate(PI/2)
        rocket_fin_r.next_to(rocket_body, RIGHT, buff=0, aligned_edge=DOWN)
        
        rocket = VGroup(rocket_body, rocket_tip, rocket_fin_l, rocket_fin_r)
        
        # Fire construction
        fire = Triangle(color=COLOR_FIRE, fill_opacity=0.8).scale(0.3).rotate(PI)
        fire.next_to(rocket_body, DOWN, buff=0)
        
        rocket_and_fire = VGroup(rocket, fire)
        self.place_at_grid(rocket_and_fire, "E4", scale_factor=0.6)
        
        self.play(FadeIn(rocket_and_fire), run_time=0.5)
        
        # Flicker updater
        def update_fire(m, dt):
            m.set_height(0.2 + 0.05 * np.sin(self.renderer.time * 20), stretch=True)
            m.next_to(rocket_body, DOWN, buff=0)
        fire.add_updater(update_fire)
        
        # Rocket launch
        self.play(rocket_and_fire.animate.move_to(self.grid["B4"]), run_time=2, rate_func=running_start)
        self.wait(0.2)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(COLOR_LINE2), run_time=0.5)
        
        # Planetary orbits
        sun = Dot(color=YELLOW).scale(2)
        orbit_path = Ellipse(width=1.5, height=0.8, color=COLOR_LINE2)
        planet = Dot(color=BLUE).scale(1.2)
        orbit_group = VGroup(sun, orbit_path, planet)
        self.place_at_grid(orbit_group, "D5", scale_factor=0.7)
        
        # Growth graph
        axes = Axes(
            x_range=[0, 3, 1], 
            y_range=[0, 3, 1], 
            x_length=1.5, 
            y_length=1.5, 
            axis_config={"include_tip": False, "color": GRAY}
        )
        graph = axes.plot(lambda x: 0.2 * np.exp(x), x_range=[0, 2.5], color=COLOR_GRAPH)
        graph_label = Text("Growth", font_size=16, color=COLOR_GRAPH)
        graph_label.next_to(axes, UP, buff=0.1)
        graph_group = VGroup(axes, graph, graph_label)
        self.place_in_area(graph_group, "B2", "C3", scale_factor=0.8)

        self.play(
            FadeIn(orbit_group),
            Create(axes),
            Create(graph),
            Write(graph_label),
            run_time=1.0
        )
        self.play(MoveAlongPath(planet, orbit_path), run_time=2, rate_func=linear)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(COLOR_LINE3), run_time=0.5)
        
        final_text = Text("The Language of Motion", color=COLOR_LINE3, font_size=28)
        self.place_in_area(final_text, "C2", "E5", scale_factor=1.0)
        
        self.play(
            FadeOut(rocket_and_fire),
            FadeOut(orbit_group),
            FadeOut(graph_group),
            FadeIn(final_text),
            run_time=1.0
        )
        self.wait(2)
