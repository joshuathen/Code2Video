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
        lecture_lines = [
            "Dandelin spheres elegantly bridge 2D and 3D geometry.",
            "This proof applies to all conic sections.",
            "These geometric principles guide orbits in space."
        ]
        self.setup_layout("Summary and Real-World Harmony", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE_A)
        
        # Ellipse and Foci to represent the 2D result of 3D construction
        # a=1.5, b=1.0 -> c = sqrt(1.5^2 - 1.0^2) = 1.118
        ellipse = Ellipse(width=3, height=2, color=BLUE_B)
        focus_dist = np.sqrt(1.5**2 - 1.0**2)
        f1 = Dot(point=LEFT * focus_dist, color=YELLOW_A)
        f2 = Dot(point=RIGHT * focus_dist, color=YELLOW_A)
        foci = VGroup(f1, f2)
        
        scene_content = VGroup(ellipse, foci)
        # Shifted to B3-F6 to avoid overlap with lecture notes as per Issue #24
        self.place_in_area(scene_content, "B3", "F6", scale_factor=0.8)
        
        self.play(Create(ellipse), run_time=1.5)
        self.play(FadeIn(foci), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(GREEN_A)
        
        # Transitioning to show it works for Parabola and Hyperbola
        # Parabola (y = x^2/p)
        parabola = ParametricFunction(
            lambda t: np.array([t, (t**2 / 2.0) - 1.0, 0]),
            t_range=[-2.0, 2.0], color=GREEN_B
        ).scale(0.6)
        parabola.move_to(ellipse.get_center())
        
        # Hyperbola (x^2/a^2 - y^2/b^2 = 1)
        hyp_l = ParametricFunction(
            lambda t: np.array([-0.8 * np.cosh(t), 1.0 * np.sinh(t), 0]),
            t_range=[-1.5, 1.5], color=RED_B
        )
        hyp_r = ParametricFunction(
            lambda t: np.array([0.8 * np.cosh(t), 1.0 * np.sinh(t), 0]),
            t_range=[-1.5, 1.5], color=RED_B
        )
        hyperbola = VGroup(hyp_l, hyp_r).scale(0.8)
        hyperbola.move_to(ellipse.get_center())

        self.play(
            ReplacementTransform(ellipse, parabola),
            FadeOut(foci),
            run_time=2
        )
        self.wait(0.5)
        self.play(
            ReplacementTransform(parabola, hyperbola),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW_A)
        
        # Real world application: Orbits
        # Back to an elliptical orbit
        orbit = Ellipse(width=3.6, height=2.4, color=BLUE_C)
        orbit.move_to(ellipse.get_center())
        
        # Planet at one focus
        oa, ob = 1.8, 1.2
        oc = np.sqrt(oa**2 - ob**2)
        planet = Dot(color="#FFFF00", radius=0.2)
        planet.move_to(orbit.get_center() + RIGHT * oc)
        planet_label = Text("Planet", font_size=16, color=YELLOW_A)
        planet_label.next_to(planet, DOWN, buff=0.1)
        
        # Satellite orbiting
        satellite = Dot(color=WHITE, radius=0.07)
        orbit_time = ValueTracker(0)
        
        def orbit_updater(m):
            t = orbit_time.get_value()
            pos = np.array([oa * np.cos(t), ob * np.sin(t), 0])
            m.move_to(orbit.get_center() + pos)
            
        satellite.add_updater(orbit_updater)

        self.play(
            FadeOut(hyperbola),
            Create(orbit),
            FadeIn(planet),
            FadeIn(planet_label),
            run_time=2
        )
        self.add(satellite)
        self.play(orbit_time.animate.set_value(2 * PI), run_time=5, rate_func=linear)
        self.wait(1)
        satellite.remove_updater(orbit_updater)
        self.wait(1)
