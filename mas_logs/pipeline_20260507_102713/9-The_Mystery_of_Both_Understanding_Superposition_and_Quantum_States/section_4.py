from manim import *
import numpy as np
import os

# Pre-emptive fix for the FileExistsError: [Errno 17] File exists: 'media/texts'
os.makedirs(os.path.join("media", "texts"), exist_ok=True)

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

class Section4Scene(TeachingScene):
    def construct(self):
        # Define 5 lecture lines for sync
        lecture_lines = [
            'Adding two valid quantum states creates another valid state.',
            'The system exists in a combination, State A+B, simultaneously.',
            'Like musical notes forming a chord, states blend together.',
            'Superposition allows states to add constructively like waves.',
            'They can also cancel out through destructive interference.'
        ]
        self.setup_layout("The Principle of Superposition", lecture_lines)

        # Colors
        COLOR_STATE_A = "#FFD700"
        COLOR_STATE_B = "#C0C0C0"
        COLOR_STATE_AB = "#FFFFFF"
        COLOR_NOTE1 = "#FF00FF"
        COLOR_NOTE2 = "#00FFFF"
        COLOR_NOTE3 = "#FFFF00"
        COLOR_CYAN = "#00FFFF"
        COLOR_ORANGE = "#FF8C00"
        COLOR_WHITE = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Two circles, 'State A' (#FFD700) and 'State B' (#C0C0C0), appear side-by-side.
        self.lecture[0].set_color(COLOR_STATE_A)
        
        circle_a = Circle(radius=0.4, color=COLOR_STATE_A, fill_opacity=0.3)
        label_a = Text("State A", font_size=16, color=COLOR_STATE_A)
        state_a_grp = VGroup(circle_a, label_a).arrange(DOWN, buff=0.1)
        self.place_at_grid(state_a_grp, "B2")
        
        circle_b = Circle(radius=0.4, color=COLOR_STATE_B, fill_opacity=0.3)
        label_b = Text("State B", font_size=16, color=COLOR_STATE_B)
        state_b_grp = VGroup(circle_b, label_b).arrange(DOWN, buff=0.1)
        self.place_at_grid(state_b_grp, "B5")
        
        self.play(FadeIn(state_a_grp), FadeIn(state_b_grp))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A third circle, 'State A+B' (#FFFFFF), appears between them via a merging animation.
        self.lecture[1].set_color(COLOR_STATE_AB)
        
        circle_ab = Circle(radius=0.5, color=COLOR_STATE_AB, fill_opacity=0.5)
        label_ab = Text("State A+B", font_size=18, color=COLOR_STATE_AB)
        state_ab_grp = VGroup(circle_ab, label_ab).arrange(DOWN, buff=0.1)
        self.place_in_area(state_ab_grp, "B3", "B4")
        
        # Ghost copies for merging effect
        ghost_a = state_a_grp.copy()
        ghost_b = state_b_grp.copy()
        
        self.play(
            ReplacementTransform(ghost_a, state_ab_grp),
            ReplacementTransform(ghost_b, state_ab_grp),
            state_a_grp.animate.set_stroke(opacity=0.2).set_fill(opacity=0.1),
            state_b_grp.animate.set_stroke(opacity=0.2).set_fill(opacity=0.1),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Three colored music note shapes move together and overlap.
        self.lecture[2].set_color(COLOR_NOTE2)
        
        def create_note(color):
            # Simple note shape: circle + line
            head = Circle(radius=0.1, color=color, fill_opacity=1)
            stem = Line(head.get_right(), head.get_right() + UP*0.3, color=color)
            return VGroup(head, stem)

        note1 = create_note(COLOR_NOTE1)
        note2 = create_note(COLOR_NOTE2)
        note3 = create_note(COLOR_NOTE3)
        
        self.place_at_grid(note1, "C1")
        self.place_at_grid(note2, "C3")
        self.place_at_grid(note3, "C5")
        
        target_center = self.grid["C3.5"] if "C3.5" in self.grid else (self.grid["C3"] + self.grid["C4"])/2
        
        self.play(FadeIn(note1), FadeIn(note2), FadeIn(note3))
        self.play(
            note1.animate.move_to(target_center),
            note2.animate.move_to(target_center),
            note3.animate.move_to(target_center),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Cyan (#00FFFF) and orange (#FF8C00) sine waves combine to create a tall white peak (#FFFFFF).
        self.lecture[3].set_color(COLOR_WHITE)
        
        # Clear previous objects to reduce clutter
        self.play(FadeOut(state_a_grp), FadeOut(state_b_grp), FadeOut(state_ab_grp), FadeOut(note1), FadeOut(note2), FadeOut(note3))
        
        phase_tracker = ValueTracker(0)
        wave_area_center = (self.grid["D1"] + self.grid["E6"]) / 2
        
        wave_cyan = FunctionGraph(
            lambda x: 0.5 * np.sin(3 * x), 
            x_range=[-2.5, 2.5], 
            color=COLOR_CYAN,
            stroke_opacity=0.4
        ).move_to(wave_area_center)
        
        wave_orange = FunctionGraph(
            lambda x: 0.5 * np.sin(3 * x), 
            x_range=[-2.5, 2.5], 
            color=COLOR_ORANGE,
            stroke_opacity=0.4
        ).move_to(wave_area_center)
        
        # Constructive peak
        sum_wave = FunctionGraph(
            lambda x: 0.5 * (np.sin(3 * x) + np.sin(3 * x + phase_tracker.get_value())),
            x_range=[-2.5, 2.5],
            color=COLOR_WHITE,
            stroke_width=4
        ).move_to(wave_area_center)

        label_c = Text("Constructive Peak", font_size=18, color=COLOR_WHITE)
        self.place_at_grid(label_c, "A3", scale_factor=0.8) # Issue 47: Moved to A3

        self.play(Create(wave_cyan), Create(wave_orange))
        self.play(Create(sum_wave), Write(label_c))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The combined white wave (#FFFFFF) ripples across the screen in a complex pattern (phase shift to destructive).
        self.lecture[4].set_color(COLOR_WHITE)
        
        label_d = Text("Destructive Cancellation", font_size=18, color=COLOR_WHITE)
        self.place_at_grid(label_d, "F3", scale_factor=0.8) # Issue 48: Moved to F3

        # Add updater for sum_wave to respond to phase_tracker
        sum_wave.add_updater(lambda m: m.become(
            FunctionGraph(
                lambda x: 0.5 * (np.sin(3 * x) + np.sin(3 * x + phase_tracker.get_value())),
                x_range=[-2.5, 2.5],
                color=COLOR_WHITE,
                stroke_width=4
            ).move_to(wave_area_center)
        ))

        self.play(
            phase_tracker.animate.set_value(PI),
            ReplacementTransform(label_c, label_d),
            run_time=3
        )
        self.wait(2)
