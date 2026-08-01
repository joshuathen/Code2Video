from manim import *

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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            'Light travels at a constant speed c in vacuum.',
            'In matter, light seems to slow down to v.',
            'We define the refractive index n as c/v.',
            'Individual photons actually always travel at speed c.',
            'Why does the macroscopic wave appear to slow down?'
        ]
        self.setup_layout("The Macroscopic Illusion vs. Microscopic Reality", lecture_lines)
        
        # Colors
        YELLOW_C = "#FFFF00"
        BLUE_C = "#ADD8E6"
        GRAY_C = "#808080"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW_C)
        
        vacuum_label = Text("Vacuum", font_size=24, color=WHITE)
        self.place_at_grid(vacuum_label, "A3", scale_factor=0.8)
        
        pulse = Dot(color=YELLOW_C, radius=0.15)
        self.place_at_grid(pulse, "B1")
        
        v_label = Text("v = c", font_size=30, color=YELLOW_C)
        v_label.add_updater(lambda m: m.next_to(pulse, UP, buff=0.1))
        
        self.add(vacuum_label, pulse, v_label)
        self.play(pulse.animate.move_to(self.grid["B6"]), run_time=2, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(BLUE_C)
        
        # Matter block
        matter_block = Rectangle(
            width=3.5, height=3.5, 
            fill_color=BLUE_C, fill_opacity=0.3, stroke_color=BLUE_C
        )
        self.place_in_area(matter_block, "C3", "F6")
        matter_text = Text("Matter", font_size=24, color=BLUE_C)
        self.place_at_grid(matter_text, "F4", scale_factor=0.8)
        
        v_slow_label = Text("v < c", font_size=30, color=BLUE_C)
        
        self.play(FadeIn(matter_block), FadeIn(matter_text))
        
        # Reset pulse for demonstration of slowdown
        self.place_at_grid(pulse, "B1")
        # Pulse enters matter
        self.play(pulse.animate.move_to(self.grid["B3"]), run_time=1, rate_func=linear)
        
        # Switch labels
        v_label.remove_updater(v_label.updaters[0])
        self.remove(v_label)
        v_slow_label.add_updater(lambda m: m.next_to(pulse, UP, buff=0.1))
        self.add(v_slow_label)
        
        # Slow down inside matter
        self.play(pulse.animate.move_to(self.grid["B6"]), run_time=2, rate_func=linear)
        v_slow_label.remove_updater(v_slow_label.updaters[0])
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW_C)
        
        formula = Text("n = c/v", font_size=42, t2c={"n": YELLOW_C})
        self.place_at_grid(formula, "D1") 
        
        self.play(Write(formula))
        self.wait(1.5)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW_C)
        
        # Clear for Microscopic view
        self.play(
            FadeOut(matter_block), FadeOut(matter_text), FadeOut(pulse), 
            FadeOut(v_slow_label), FadeOut(vacuum_label), FadeOut(formula)
        )
        
        atoms = VGroup()
        for r in ["C", "D", "E"]:
            for c in ["2", "3", "4", "5"]:
                atom = Dot(color=GRAY_C, radius=0.1)
                pos = self.grid[f"{r}{c}"] + np.array([np.random.uniform(-0.15,0.15), np.random.uniform(-0.15,0.15), 0])
                atom.move_to(pos)
                atoms.add(atom)
        
        photons = VGroup(*[
            Dot(color=YELLOW_C, radius=0.06).move_to(self.grid["D1"] + UP*i*0.3)
            for i in range(-2, 3)
        ])
        
        photon_speed_label = Text("Speed is always c", font_size=20, color=YELLOW_C)
        self.place_at_grid(photon_speed_label, "A3", scale_factor=0.8)

        self.play(Create(atoms), FadeIn(photons), Write(photon_speed_label))
        
        # Photons move across at constant speed c
        photon_moves = [
            p.animate.move_to(p.get_center() + RIGHT * 5)
            for p in photons
        ]
        self.play(*photon_moves, run_time=2, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(BLUE_C)
        
        self.play(FadeOut(atoms), FadeOut(photons), FadeOut(photon_speed_label))
        
        phase_tracker = ValueTracker(0)
        
        # Vacuum wave (Top)
        vac_wave = always_redraw(lambda: ParametricFunction(
            lambda t: np.array([t, 0.4 * np.sin(2 * PI * (t - phase_tracker.get_value())), 0]),
            t_range=[0.5, 5.5],
            color=YELLOW_C
        ).shift(UP * 1.0))
        
        # Matter wave (Bottom) - Phase lags behind
        mat_wave = always_redraw(lambda: ParametricFunction(
            lambda t: np.array([t, 0.4 * np.sin(2 * PI * (t - 0.7 * phase_tracker.get_value())), 0]),
            t_range=[0.5, 5.5],
            color=BLUE_C
        ).shift(DOWN * 1.0))
        
        vac_wave_label = Text("Vacuum Wave", font_size=18, color=YELLOW_C)
        self.place_at_grid(vac_wave_label, "A1", scale_factor=0.8)
        vac_wave_label.shift(UP*1.0)
        
        mat_wave_label = Text("Collective Wave (Appears Slow)", font_size=18, color=BLUE_C)
        self.place_at_grid(mat_wave_label, "D1", scale_factor=0.8)
        mat_wave_label.shift(DOWN*1.0)

        self.play(Create(vac_wave), Create(mat_wave), FadeIn(vac_wave_label), FadeIn(mat_wave_label))
        self.play(phase_tracker.animate.set_value(3), run_time=4, rate_func=linear)
        self.wait(1)
