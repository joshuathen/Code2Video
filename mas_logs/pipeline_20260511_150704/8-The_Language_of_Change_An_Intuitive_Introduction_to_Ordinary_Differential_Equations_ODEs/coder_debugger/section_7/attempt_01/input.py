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

class Section7Scene(TeachingScene):
    def construct(self):
        # Initialize layout
        lines = [
            'ODEs govern everything from orbits to cooling coffee.',
            'They bridge physical laws and future predictions.',
            'Master this language to understand the world.'
        ]
        self.setup_layout("Summary & Real-World Scope", lines)
        
        # Internal time tracker for updaters
        time_tracker = ValueTracker(0)
        self.add(time_tracker)
        time_tracker.add_updater(lambda m, dt: m.increment_value(dt))

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # 1. Pendulum Visual
        pend_pivot_marker = Dot(radius=0, fill_opacity=0)
        pend_string = Line(ORIGIN, DOWN*0.5, color=WHITE)
        pend_bob = Dot(color=WHITE, radius=0.08).move_to(DOWN*0.5)
        pendulum = VGroup(pend_pivot_marker, pend_string, pend_bob)

        # 2. Orbit Visual
        orbit_path = Circle(radius=0.35, color=WHITE, stroke_width=1)
        sun = Dot(color=YELLOW, radius=0.08)
        planet = Dot(color=BLUE, radius=0.04)
        orbit_sub = VGroup(orbit_path, sun, planet)

        # 3. Cooling Cup Visual
        cup = RoundedRectangle(height=0.4, width=0.3, color=WHITE)
        steam = VGroup(*[Line(ORIGIN, UP*0.1, color=WHITE).shift(RIGHT*x) for x in [-0.06, 0, 0.06]])
        steam.next_to(cup, UP, buff=0.05)
        cup_sub = VGroup(cup, steam)

        # Montage of physical systems
        visual_montage = VGroup(pendulum, orbit_sub, cup_sub).arrange(RIGHT, buff=0.8)
        
        # Add Updaters after positioning
        def update_pendulum(m):
            pivot = m[0].get_center()
            angle = 0.4 * np.sin(time_tracker.get_value() * 3)
            end = pivot + np.array([0.5 * np.sin(angle), -0.5 * np.cos(angle), 0])
            m[1].set_points_as_corners([pivot, end])
            m[2].move_to(end)
        pendulum.add_updater(update_pendulum)

        def update_planet(m):
            m.move_to(
                orbit_path.get_center() + np.array([
                    0.35 * np.cos(time_tracker.get_value() * 4),
                    0.35 * np.sin(time_tracker.get_value() * 4),
                    0
                ])
            )
        planet.add_updater(update_planet)
        
        # ODE overlays [Color: #00FFFF]
        ode_p = MathTex(r"\theta'' + \frac{g}{L}\theta = 0", color="#00FFFF", font_size=14).next_to(pendulum, DOWN, buff=0.1)
        ode_o = MathTex(r"F = G\frac{Mm}{r^2}", color="#00FFFF", font_size=14).next_to(orbit_sub, DOWN, buff=0.1)
        ode_c = MathTex(r"\frac{dT}{dt} = -k(T-T_s)", color="#00FFFF", font_size=14).next_to(cup_sub, DOWN, buff=0.1)
        
        orbit_group = VGroup(visual_montage, ode_p, ode_o, ode_c)
        
        # Resolve Issue 51: Position visual montage/orbit_group in upper area
        self.place_in_area(orbit_group, 'A2', 'C5', scale_factor=1.2)
        
        self.play(FadeIn(orbit_group))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(CYAN)
        )
        
        # Bridge logic Visual [Matching CYAN of ODEs]
        bridge_arrow = Arrow(LEFT*1.2, RIGHT*1.2, color=CYAN)
        law_text = Text("Physical Law", font_size=16, color=WHITE).next_to(bridge_arrow, LEFT, buff=0.2)
        pred_text = Text("Prediction", font_size=16, color=WHITE).next_to(bridge_arrow, RIGHT, buff=0.2)
        bridge_logic_group = VGroup(law_text, bridge_arrow, pred_text)
        
        # Resolve Issue 52: Position bridge logic in lower area
        self.place_in_area(bridge_logic_group, 'D3', 'F6', scale_factor=0.9)
        
        self.play(Create(bridge_logic_group))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(WHITE)
        )
        
        # Final transition to title card
        final_title = Text("The Language of Change", font_size=36, color=WHITE)
        
        self.play(
            FadeOut(orbit_group),
            FadeOut(bridge_logic_group),
            Write(self.place_at_grid(final_title, "C4", scale_factor=1.0))
        )
        self.wait(3)
