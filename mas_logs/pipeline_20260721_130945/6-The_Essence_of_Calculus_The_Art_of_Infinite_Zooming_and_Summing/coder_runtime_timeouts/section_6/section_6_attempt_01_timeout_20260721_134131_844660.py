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
        self.play(self.lecture[0].animate.set_color(COLOR_LINE1))

        # Rocket construction
        rocket_body = Rectangle(width=0.4, height=1.0, color=WHITE, fill_opacity=1)
        rocket_tip = Triangle(color=WHITE, fill_opacity=1).scale(0.2)
        rocket_tip.next_to(rocket_body, UP, buff=0)
        # Procedural fins using Polygon
        rocket_fin_l = Polygon([-0.2, -0.4, 0], [-0.4, -0.6, 0], [-0.2, -0.6, 0], color=WHITE, fill_opacity=1)
        rocket_fin_r = Polygon([0.2, -0.4, 0], [0.4, -0.6, 0], [0.2, -0.6, 0], color=WHITE, fill_opacity=1)
        
        rocket = VGroup(rocket_body, rocket_tip, rocket_fin_l, rocket_fin_r)
        
        # Fire construction
        fire = Triangle(color=COLOR_FIRE, fill_opacity=0.8).scale(0.3).rotate(PI)
        fire.next_to(rocket_body, DOWN, buff=0)
        
        rocket_and_fire = VGroup(rocket, fire)
        # Start at F4 (bottom of grid, Col 4 to avoid crowding notes)
        self.place_at_grid(rocket_and_fire, "F4", scale_factor=0.8)
        
        self.play(FadeIn(rocket_and_fire))
        
        # Simple flicker updater for the fire
        def update_fire(m, dt):
            # Scale height slightly for flicker effect
            new_height = 0.3 + 0.1 * np.sin(self.renderer.time * 20)
            m.set_height(new_height, stretch=True)
            m.next_to(rocket_body, DOWN, buff=0)
        fire.add_updater(update_fire)
        
        # Rocket launch - moves up to Row B
        self.play(rocket_and_fire.animate.move_to(self.grid["B4"]), run_time=3, rate_func=running_start)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Highlight Line 2
        self.play(self.lecture[1].animate.set_color(COLOR_LINE2))
        
        # Planetary orbits
        sun = Dot(color=YELLOW).scale(2)
        orbit_path = Ellipse(width=2.5, height=1.2, color=COLOR_LINE2)
        planet = Dot(color=BLUE).scale(1.2)
        orbit_group = VGroup(sun, orbit_path, planet)
        # Position orbits in a separate area from the rocket
        self.place_in_area(orbit_group, "B5", "D6", scale_factor=0.7)
        
        # Growth graph representing medicine/population spread
        axes = Axes(
            x_range=[0, 4, 1], 
            y_range=[0, 4, 1], 
            x_length=2.5, 
            y_length=2, 
            axis_config={"include_tip": False, "color": GRAY}
        )
        graph = axes.plot(lambda x: 0.1 * np.exp(x), x_range=[0, 3.5], color=COLOR_GRAPH)
        graph_label = Text("Growth", font_size=20, color=COLOR_GRAPH)
        graph_label.next_to(axes, UP, buff=0.1)
        graph_group = VGroup(axes, graph, graph_label)
        # Position graph in a separate area
        self.place_in_area(graph_group, "B2", "D3", scale_factor=0.7)

        # Planet movement updater
        def update_planet(m, dt):
            m.move_to(orbit_path.point_at_angle(self.renderer.time))
        planet.add_updater(update_planet)

        self.play(
            FadeIn(orbit_group),
            Create(axes),
            Create(graph),
            Write(graph_label)
        )
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # Highlight Line 3
        self.play(self.lecture[2].animate.set_color(COLOR_LINE3))
        
        final_text = Text("The Language of Motion", color=COLOR_LINE3, font_size=32)
        # Final text area center (positioned prominently across the right area)
        self.place_in_area(final_text, "C2", "E6", scale_factor=1.2)
        
        # Cleanup animation elements and fade in final text for conclusion
        self.play(
            FadeOut(rocket_and_fire),
            FadeOut(orbit_group),
            FadeOut(graph_group),
            FadeIn(final_text)
        )
        self.wait(3)
