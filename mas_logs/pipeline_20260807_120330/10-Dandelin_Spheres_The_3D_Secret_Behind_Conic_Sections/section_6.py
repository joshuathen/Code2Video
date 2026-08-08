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
        # Data from storyboard
        title = "Application: From Ice Cones to Orbits"
        lines = [
            "Tilting the plane creates other conic sections.",
            "This geometry governs how planets orbit stars.",
            "3D shapes explain our 2D mathematical world."
        ]
        self.setup_layout(title, lines)

        # Colors from animation description
        color_parabola = "#FF00FF"
        color_hyperbola = "#00FF00"
        color_planet = "#ADD8E6"
        color_sun = "#FFFF00"
        color_text = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Representing a cone via cross-section
        # Using columns 2-6 to maintain gap from lecture (B021)
        cone_l = Line(self.grid["B4"], self.grid["F2"], color=GREY_B)
        cone_r = Line(self.grid["B4"], self.grid["F6"], color=GREY_B)
        cone = VGroup(cone_l, cone_r)
        
        # Slicing Plane line
        plane = Line(self.grid["D3"], self.grid["D5"], color=WHITE)
        plane_label = Text("Plane", font_size=18).next_to(plane, UP, buff=0.1)
        
        self.play(Create(cone), Create(plane), Write(plane_label))
        self.wait(1)
        
        # Parabola Animation
        parabola = FunctionGraph(
            lambda x: 0.8 * x**2 - 0.5, 
            x_range=[-1.2, 1.2], 
            color=color_parabola
        )
        self.place_in_area(parabola, "C3", "E5")
        
        # Issue 35: Fix Parabola label position to B4
        parabola_label = Text("Parabola", color=color_parabola, font_size=24)
        self.place_at_grid(parabola_label, "B4", scale_factor=0.8)
        
        self.play(
            Rotate(plane, angle=PI/6),
            FadeIn(parabola),
            Write(parabola_label)
        )
        self.wait(1)
        
        # Hyperbola Animation
        hyperbola_top = FunctionGraph(
            lambda x: np.sqrt(x**2 + 0.3), 
            x_range=[-1, 1], 
            color=color_hyperbola
        )
        hyperbola_bot = FunctionGraph(
            lambda x: -np.sqrt(x**2 + 0.3), 
            x_range=[-1, 1], 
            color=color_hyperbola
        )
        hyperbola = VGroup(hyperbola_top, hyperbola_bot)
        self.place_in_area(hyperbola, "C3", "E5")
        
        # Issue 36: Fix Hyperbola label position to B4
        hyperbola_label = Text("Hyperbola", color=color_hyperbola, font_size=24)
        self.place_at_grid(hyperbola_label, "B4", scale_factor=0.8)
        
        self.play(
            Rotate(plane, angle=PI/6),
            FadeOut(parabola),
            FadeOut(parabola_label),
            FadeIn(hyperbola),
            Write(hyperbola_label)
        )
        self.wait(2)
        
        # Cleanup for transition
        self.play(
            FadeOut(cone), FadeOut(plane), FadeOut(plane_label),
            FadeOut(hyperbola), FadeOut(hyperbola_label),
            self.lecture[0].animate.set_color(WHITE)
        )

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        
        # Orbital system
        # Center of ellipse at D4, Sun at focus D3
        orbit_path = Ellipse(width=3.0, height=2.236, color=BLUE_E).set_stroke(opacity=0.6)
        self.place_at_grid(orbit_path, "D4")
        
        sun = Dot(color=color_sun, radius=0.2)
        self.place_at_grid(sun, "D3") # Focus
        sun_label = Text("Sun", color=color_sun, font_size=20).next_to(sun, DOWN, buff=0.1)
        
        planet = Dot(color=color_planet, radius=0.12)
        planet_label = Text("Planet", color=color_planet, font_size=18)
        
        # Setup orbital movement (B010: Use persistent mobjects + ValueTracker)
        orbit_tracker = ValueTracker(0)
        planet.add_updater(lambda m: m.move_to(orbit_path.point_from_proportion(orbit_tracker.get_value() % 1)))
        planet_label.add_updater(lambda m: m.next_to(planet, UR, buff=0.05))
        
        self.play(Create(orbit_path), FadeIn(sun), Write(sun_label))
        self.add(planet, planet_label)
        self.play(orbit_tracker.animate.set_value(1.5), run_time=5, rate_func=linear)
        self.wait(1)
        
        # Transition out
        self.play(
            FadeOut(orbit_path), FadeOut(sun), FadeOut(sun_label),
            FadeOut(planet), FadeOut(planet_label),
            self.lecture[1].animate.set_color(WHITE)
        )

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        
        # Issue 34: Final message 'The Geometry of the Universe' at B3-E6, scale 0.8
        final_msg = Text("The Geometry of the Universe", color=color_text, font_size=32)
        self.place_in_area(final_msg, "B3", "E6", scale_factor=0.8)
        
        self.play(Write(final_msg))
        self.play(final_msg.animate.scale(1.1), run_time=1)
        self.play(final_msg.animate.scale(1/1.1), run_time=1)
        self.wait(2)
        
        # End of section
        self.play(FadeOut(final_msg), self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
