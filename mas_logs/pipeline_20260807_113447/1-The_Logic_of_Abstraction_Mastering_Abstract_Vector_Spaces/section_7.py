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
        # Title and Lecture lines from storyboard
        title_text = "Summary & The Power of Generalization"
        lecture_lines = [
            "One tool solves diverse scientific problems.",
            "Physics and AI share underlying math.",
            "Power lies in rules, not objects.",
            "Generalization reveals deep structural patterns.",
            "Mastery starts with logical abstraction."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        PHYSICS_COLOR = WHITE
        AI_COLOR = BLUE_C
        AUDIO_COLOR = GREEN_C
        AXIOM_COLOR = YELLOW_A
        
        # === Animation for Lecture Line 1 ===
        # Split screen shows Physics, AI, and Audio panels.
        
        physics_panel_rect = Rectangle(width=1.8, height=5.5, color=WHITE, stroke_width=2, fill_opacity=0.1)
        ai_panel_rect = Rectangle(width=1.8, height=5.5, color=BLUE, stroke_width=2, fill_opacity=0.1)
        audio_panel_rect = Rectangle(width=1.8, height=5.5, color=GREEN, stroke_width=2, fill_opacity=0.1)
        
        self.place_in_area(physics_panel_rect, "A1", "F2")
        self.place_in_area(ai_panel_rect, "A3", "F4")
        self.place_in_area(audio_panel_rect, "A5", "F6")
        
        physics_label = Text("Physics", font_size=20, color=WHITE)
        ai_label = Text("AI", font_size=20, color=BLUE)
        audio_label = Text("Audio", font_size=20, color=GREEN)
        
        self.place_at_grid(physics_label, "A1", scale_factor=0.8)
        self.place_at_grid(ai_label, "A3", scale_factor=0.8)
        self.place_at_grid(audio_label, "A5", scale_factor=0.8)
        
        self.lecture[0].set_color(YELLOW)
        self.play(
            Create(physics_panel_rect),
            Create(ai_panel_rect),
            Create(audio_panel_rect),
            Write(physics_label),
            Write(ai_label),
            Write(audio_label),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Physics and AI share underlying math.
        # Physics panel shows a white planet orbiting a sun.
        # AI panel shows a grid of blue facial points.
        
        # Physics: Sun and Orbiting Planet
        sun = Dot(color=YELLOW).scale(2)
        self.place_in_area(sun, "C1", "D2")
        orbit = Circle(radius=0.7, color=GRAY, stroke_width=1).move_to(sun.get_center())
        planet = Dot(color=WHITE, radius=0.1)
        
        orbit_tracker = ValueTracker(0)
        planet.add_updater(lambda m: m.move_to(
            orbit.point_from_proportion(orbit_tracker.get_value() % 1)
        ))
        
        # AI: Facial points grid
        ai_points = VGroup()
        for x_off in np.linspace(-0.6, 0.6, 4):
            for y_off in np.linspace(-0.8, 0.8, 5):
                dot = Dot(radius=0.06, color=BLUE_C)
                dot.move_to(ai_panel_rect.get_center() + np.array([x_off, y_off, 0]))
                ai_points.add(dot)
        
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        self.play(
            FadeIn(sun),
            Create(orbit),
            FadeIn(planet),
            FadeIn(ai_points, shift=UP),
            run_time=1.5
        )
        self.play(orbit_tracker.animate.set_value(1), run_time=2, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Power lies in rules, not objects.
        # Audio panel shows a green oscillating sound wave.
        
        wave_tracker = ValueTracker(0)
        audio_wave = VMobject(color=GREEN_C, stroke_width=3)
        def update_wave(m):
            points = [
                audio_panel_rect.get_center() + np.array([x, 0.6 * np.sin(3 * PI * (x - wave_tracker.get_value())), 0])
                for x in np.linspace(-0.8, 0.8, 40)
            ]
            m.set_points_as_corners(points)
            
        audio_wave.add_updater(update_wave)
        
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        self.play(
            FadeIn(audio_wave),
            orbit_tracker.animate.set_value(2),
            wave_tracker.animate.set_value(1),
            run_time=2,
            rate_func=linear
        )
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        # Generalization reveals deep structural patterns.
        # Yellow Axiom symbols overlay all three panels.
        
        axioms = VGroup(
            MathTex(r"\vec{u} + \vec{v} = \vec{v} + \vec{u}", color=YELLOW, font_size=28),
            MathTex(r"c(\vec{u} + \vec{v}) = c\vec{u} + c\vec{v}", color=YELLOW, font_size=28),
            MathTex(r"1\vec{v} = \vec{v}", color=YELLOW, font_size=28)
        )
        
        self.place_in_area(axioms[0], "C1", "D2") 
        self.place_in_area(axioms[1], "C3", "D4") 
        self.place_in_area(axioms[2], "C5", "D6") 
        
        axiom_bgs = VGroup(*[
            BackgroundRectangle(ax, color=BLACK, fill_opacity=0.8, buff=0.1)
            for ax in axioms
        ])
        
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        self.play(
            FadeIn(axiom_bgs),
            Write(axioms),
            orbit_tracker.animate.set_value(3),
            wave_tracker.animate.set_value(2),
            run_time=2,
            rate_func=linear
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Mastery starts with logical abstraction.
        
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        self.play(
            axioms.animate.scale(1.2).set_color(YELLOW_A),
            orbit_tracker.animate.set_value(5),
            wave_tracker.animate.set_value(4),
            run_time=4,
            rate_func=linear
        )
        self.wait(2)
