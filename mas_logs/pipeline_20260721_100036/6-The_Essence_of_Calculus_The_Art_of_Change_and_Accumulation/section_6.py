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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup layout with title and lecture lines
        self.setup_layout("Calculus in Action: The Modern World", [
            "- Calculus is the engine powering modern physics and engineering.",
            "- It guides planets in orbit and rockets landing safely.",
            "- We use it to model anything that moves or changes."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Show a rocket icon (#FFFF00) and an engine diagram (#00FFFF). Change lecture line 1 color to #FFFF00.
        
        # Rocket Icon
        rocket_body = Rectangle(width=0.3, height=0.6, fill_opacity=1, color="#FFFF00")
        rocket_tip = Triangle(fill_opacity=1, color="#FFFF00").scale(0.15).next_to(rocket_body, UP, buff=0)
        rocket_fin_l = Triangle(fill_opacity=1, color="#FFFF00").scale(0.1).rotate(PI/2).next_to(rocket_body, LEFT, buff=0).shift(DOWN*0.2)
        rocket_fin_r = Triangle(fill_opacity=1, color="#FFFF00").scale(0.1).rotate(-PI/2).next_to(rocket_body, RIGHT, buff=0).shift(DOWN*0.2)
        rocket = VGroup(rocket_body, rocket_tip, rocket_fin_l, rocket_fin_r)
        # Resolved Issue 42: B2 -> B3 to avoid crowding the lecture area
        self.place_at_grid(rocket, "B3", scale_factor=0.8)
        
        # Engine Diagram (Simple abstract representation)
        engine_circle = Circle(radius=0.4, color="#00FFFF", stroke_width=2)
        engine_core = Dot(color="#00FFFF").scale(0.5)
        engine_rays = VGroup(*[Line(ORIGIN, 0.4*RIGHT, color="#00FFFF").rotate(a) for a in np.linspace(0, 2*PI, 12)])
        engine = VGroup(engine_circle, engine_core, engine_rays)
        self.place_at_grid(engine, "B5", scale_factor=0.8)
        
        self.play(
            self.lecture[0].animate.set_color("#FFFF00"),
            FadeIn(rocket, shift=UP),
            Create(engine),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show planetary orbits (#00FF00) and the rocket landing sequence. Change lecture line 2 color to #FFFF00.
        
        # Orbits
        orbit_a = Ellipse(width=2.0, height=1.0, color="#00FF00").set_stroke(opacity=0.5)
        planet_a = Dot(color=WHITE, radius=0.08).move_to(orbit_a.point_at_angle(0))
        orbit_b = Ellipse(width=1.2, height=0.6, color="#00FF00").rotate(PI/3).set_stroke(opacity=0.5)
        planet_b = Dot(color=WHITE, radius=0.06).move_to(orbit_b.point_at_angle(PI/2))
        orbits_group = VGroup(orbit_a, planet_a, orbit_b, planet_b)
        # Resolved Issue 41: D1-F3 -> D2-F4 to avoid crowding the lecture area
        self.place_in_area(orbits_group, "D2", "F4")
        
        self.play(
            self.lecture[1].animate.set_color("#FFFF00"),
            Create(orbits_group),
            run_time=1
        )
        
        # Landing and Orbit animation
        self.play(
            MoveAlongPath(planet_a, orbit_a),
            MoveAlongPath(planet_b, orbit_b),
            rocket.animate.move_to(self.grid["F5"]).rotate(PI), # Descend and flip for landing
            run_time=4,
            rate_func=smooth
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Flash icons for 'Weather', 'Stock Market', and 'Robotics' in #FF8C00. Change lecture line 3 color to #FFFF00.
        
        # Weather Icon
        weather_icon = VGroup(
            Circle(radius=0.15, color="#FF8C00", fill_opacity=1).shift(LEFT*0.1),
            Circle(radius=0.2, color="#FF8C00", fill_opacity=1).shift(UP*0.1),
            Circle(radius=0.15, color="#FF8C00", fill_opacity=1).shift(RIGHT*0.1)
        )
        # Resolved Issue 40: C1 -> C2 to avoid crowding the lecture area
        self.place_at_grid(weather_icon, "C2", scale_factor=0.8)
        
        # Stock Market Icon
        stock_line = VMobject(color="#FF8C00", stroke_width=4)
        stock_line.set_points_as_corners([LEFT*0.3+DOWN*0.2, LEFT*0.1+UP*0.1, RIGHT*0.1+DOWN*0.1, RIGHT*0.3+UP*0.3])
        self.place_at_grid(stock_line, "C3", scale_factor=0.8)
        
        # Robotics Icon
        robotic_arm = VGroup(
            Line(ORIGIN, UP*0.3, color="#FF8C00", stroke_width=5),
            Line(UP*0.3, UP*0.3+RIGHT*0.3, color="#FF8C00", stroke_width=5),
            Dot(UP*0.3+RIGHT*0.3, color="#FF8C00")
        )
        self.place_at_grid(robotic_arm, "C5", scale_factor=0.8)
        
        icons = VGroup(weather_icon, stock_line, robotic_arm)
        
        self.play(
            self.lecture[2].animate.set_color("#FFFF00"),
            FadeIn(icons, scale=1.5),
            run_time=2
        )
        self.wait(3)
