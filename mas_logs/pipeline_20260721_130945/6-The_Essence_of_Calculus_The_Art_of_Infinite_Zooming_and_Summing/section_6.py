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
        COLOR_HIGHLIGHT = WHITE
        COLOR_CYAN = "#00FFFF"
        COLOR_RED = "#FF0000"
        COLOR_YELLOW = "#FFFF00"
        COLOR_FIRE = "#FF4500"

        # === Animation for Lecture Line 1 ===
        # Highlight Line 1
        self.play(self.lecture[0].animate.set_color(COLOR_HIGHLIGHT), run_time=0.4)

        # Simplified Rocket Construction
        rocket_body = Rectangle(width=0.3, height=0.6, color=WHITE, fill_opacity=1)
        rocket_nose = Triangle(color=WHITE, fill_opacity=1).scale(0.15).next_to(rocket_body, UP, buff=0)
        rocket_fire = Triangle(color=COLOR_FIRE, fill_opacity=0.8).scale(0.2).rotate(PI).next_to(rocket_body, DOWN, buff=0)
        
        rocket_group = VGroup(rocket_body, rocket_nose, rocket_fire)
        # Fix for Issue 43: Moving initial definition to A5 as requested, but shifting down for launch
        self.place_at_grid(rocket_group, 'A5', scale_factor=0.8)
        rocket_group.shift(DOWN * 5) # Prepare for launch from bottom
        
        # Efficient flicker updater using self.renderer.time
        def update_fire(m):
            # L008: Use set_fill_opacity for stability
            m.set_fill_opacity(0.6 + 0.4 * np.abs(np.sin(self.renderer.time * 20)))

        rocket_fire.add_updater(update_fire)
        
        self.play(FadeIn(rocket_group), run_time=0.5)
        # Rocket launch motion - target is A5 to resolve Issue 43 fully
        self.play(rocket_group.animate.move_to(self.grid["A5"]), run_time=1.4, rate_func=rush_into)

        # === Animation for Lecture Line 2 ===
        # Highlight Line 2 in Cyan to match orbit
        self.play(self.lecture[1].animate.set_color(COLOR_CYAN), run_time=0.4)
        
        # Planetary orbits
        sun = Dot(color=YELLOW).scale(1.2)
        orbit = Ellipse(width=1.0, height=0.5, color=COLOR_CYAN)
        planet = Dot(color=BLUE).scale(0.7).move_to(orbit.point_from_proportion(0))
        orbit_sys = VGroup(sun, orbit, planet)
        # Fix for Issue 45: Move orbit_sys to F5 to utilize grid corners and avoid clustering
        self.place_at_grid(orbit_sys, 'F5', scale_factor=0.8)
        
        # Growth graph (simplified VMobject for performance)
        axes = Axes(
            x_range=[0, 2], y_range=[0, 2], 
            x_length=1.2, y_length=1.2, 
            axis_config={"color": GRAY, "include_tip": False, "include_ticks": False}
        )
        curve_pts = [axes.c2p(x, 0.2 * np.exp(x)) for x in np.linspace(0, 1.8, 15)]
        curve = VMobject(color=COLOR_RED).set_points_as_corners(curve_pts)
        graph = VGroup(axes, curve)
        # Fix for Issue 44: Move graph to A2 to avoid obstruction by final_msg
        self.place_at_grid(graph, 'A2', scale_factor=0.8)

        self.play(
            FadeIn(orbit_sys),
            Create(graph),
            run_time=0.8
        )
        # Animate orbit
        self.play(MoveAlongPath(planet, orbit), run_time=1.2, rate_func=linear)

        # === Animation for Lecture Line 3 ===
        # Highlight Line 3 in Yellow to match message
        self.play(self.lecture[2].animate.set_color(COLOR_YELLOW), run_time=0.4)
        
        final_msg = Text("The Language of Motion", color=COLOR_YELLOW, font_size=32)
        # Position in a clear area, avoiding Col 1 (lecture) and corner objects
        self.place_in_area(final_msg, "B2", "E6", scale_factor=0.9)
        
        # Clean up updaters
        rocket_fire.clear_updaters()
        
        self.play(
            FadeOut(rocket_group),
            FadeOut(orbit_sys),
            FadeOut(graph),
            FadeIn(final_msg),
            run_time=1.0
        )
        self.wait(2.0)
