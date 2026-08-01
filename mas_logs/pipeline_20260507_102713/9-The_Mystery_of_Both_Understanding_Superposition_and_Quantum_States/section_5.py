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
        # Initialize layout with updated lines
        lines = [
            'A quantum system exists as a swarm of possibilities.', 
            'Observing the system triggers an immediate change.', 
            'Superposition collapses into a single, definite state.', 
            'This transition is known as the state collapse.', 
            'The original quantum blur is lost upon measurement.'
        ]
        self.setup_layout("Measurement and State Collapse", lines)

        # Value tracker for swarm movement
        time_tracker = ValueTracker(0)
        self.add(time_tracker)
        time_tracker.add_updater(lambda m, dt: m.increment_value(dt))

        # Colors
        COLOR_SWARM = "#CCCCCC"
        COLOR_EYE = "#FFFFFF"
        COLOR_COLLAPSE = "#FF0000"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_SWARM)
        
        # Create swarm of faint white particles
        swarm = VGroup(*[Dot(radius=0.04, color=COLOR_SWARM, fill_opacity=0.6) for _ in range(40)])
        for p in swarm:
            # Spread randomly in a box area defined by the grid roughly (B2 to E5 region)
            p.move_to(np.array([
                np.random.uniform(1.5, 4.5),
                np.random.uniform(-1.5, 1.5),
                0
            ]))
            p.initial_pos = p.get_center()
            p.phase = np.random.random() * TAU
            # Add subtle floating movement
            p.add_updater(lambda m, p=p: m.move_to(p.initial_pos + np.array([
                0.12 * np.sin(time_tracker.get_value() * 1.5 + p.phase),
                0.12 * np.cos(time_tracker.get_value() * 2.1 + p.phase),
                0
            ])))

        self.play(FadeIn(swarm))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(COLOR_EYE)
        
        # Load Eye icon asset
        eye = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/eye.svg")
        eye.set_color(COLOR_EYE)
        self.place_at_grid(eye, 'A4', scale_factor=0.7)

        self.play(FadeIn(eye))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(COLOR_COLLAPSE)
        
        # Single solid red circle
        red_circle = Dot(radius=0.2, color=COLOR_COLLAPSE)
        self.place_at_grid(red_circle, 'C4')

        # Snap: Swarm disappears, red circle appears
        self.play(
            FadeOut(swarm),
            FadeIn(red_circle),
            run_time=0.4
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(COLOR_COLLAPSE)
        
        # Use Area placement to avoid overlap and improve layout
        collapse_text = Text("State Collapse", font_size=24, color=COLOR_COLLAPSE)
        self.place_in_area(collapse_text, 'C5', 'D6', scale_factor=0.8)
        
        self.play(Write(collapse_text))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(COLOR_EYE)
        
        # Pulse animation for the Eye asset
        self.play(
            eye.animate.scale(1.2),
            rate_func=there_and_back,
            run_time=0.8
        )
        self.play(
            eye.animate.scale(1.2),
            rate_func=there_and_back,
            run_time=0.8
        )
        self.wait(2)
