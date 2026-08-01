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

class Section2Scene(TeachingScene):
    def construct(self):
        # Setup layout
        lecture_lines = [
            "Slowing down is an illusion created by wave superposition.",
            "Incoming waves force electrons to oscillate like springs.",
            "These oscillating electrons radiate their own secondary waves.",
            "Secondary waves interfere with the original driving wave.",
            "The resultant wave appears delayed, moving slower through matter."
        ]
        self.setup_layout("The Microscopic Origin: The Atomic Dance", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Show a row of atoms (#FFFFFF) as circles on [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/springs.svg]
        self.lecture[0].set_color(WHITE)
        
        atom_row = VGroup()
        nuclei = VGroup()
        springs = VGroup()
        electrons = VGroup()
        
        for i in range(1, 7):
            # Issue 31: Asset Integration for Lorentz Oscillator Model
            spring_asset = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/springs.svg").scale(0.3)
            nucleus = Circle(radius=0.08, color=WHITE, fill_opacity=1).move_to(spring_asset.get_bottom())
            electron = Dot(color=YELLOW, radius=0.1).move_to(spring_asset.get_top())
            
            atom_unit = VGroup(nucleus, spring_asset, electron)
            atom_unit.shift(RIGHT * (i - 1) * 0.8) 
            
            atom_row.add(atom_unit)
            nuclei.add(nucleus)
            springs.add(spring_asset)
            electrons.add(electron)

        # Issue 37: Better positioning to avoid cramped flow
        self.place_in_area(atom_row, 'D1', 'D6', scale_factor=0.8)

        self.play(Create(atom_row), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # An incoming light wave (#00FFFF) hits the atoms, making the electrons (#FFFF00) oscillate.
        self.lecture[1].set_color("#00FFFF")
        
        time_tracker = ValueTracker(0)
        wave_freq = 1.5
        wave_speed = 2.0
        
        # Issue 38: Driving wave bounded to grid
        incoming_wave = always_redraw(lambda: self.place_in_area(
            FunctionGraph(
                lambda x: 0.4 * np.sin(wave_freq * (x - wave_speed * time_tracker.get_value())),
                x_range=[0, 5],
                color="#00FFFF"
            ), 'B1', 'E6', scale_factor=1.0)
        )

        self.play(Create(incoming_wave))
        
        # Electrons oscillate with the wave
        for i in range(len(electrons)):
            # Capture initial relative resting vector between electron and nucleus
            resting_vec = electrons[i].get_center() - nuclei[i].get_center()
            
            electrons[i].add_updater(lambda m, i=i, rv=resting_vec: m.move_to(
                nuclei[i].get_center() + rv + UP * 0.4 * np.sin(
                    wave_freq * (nuclei[i].get_center()[0] - wave_speed * time_tracker.get_value())
                )
            ))
            
            # Spring stretches to follow electron oscillation
            springs[i].add_updater(lambda m, i=i: m.stretch_to_fit_height(
                max(0.1, np.linalg.norm(electrons[i].get_center() - nuclei[i].get_center()))
            ).move_to((electrons[i].get_center() + nuclei[i].get_center()) / 2))

        self.play(time_tracker.animate.set_value(2), run_time=2, rate_func=linear)
        
        # === Animation for Lecture Line 3 ===
        # Show the oscillation spreading through the atoms with a slight delay between each.
        self.lecture[2].set_color("#FFFF00")
        self.play(time_tracker.animate.set_value(4), run_time=2, rate_func=linear)

        # === Animation for Lecture Line 4 ===
        # Visualize secondary EM waves (#00FF00) being emitted by each oscillating electron.
        self.lecture[3].set_color("#00FF00")
        
        def create_secondary_ripples():
            ripples = VGroup()
            t = time_tracker.get_value()
            for i in range(len(electrons)):
                # Each electron emits a pulse with a phase delay
                phase = i * 0.5
                r = ((t * 1.2 - phase) % 2) * 0.4
                if r > 0.05:
                    ripple = Circle(
                        radius=r,
                        color="#00FF00",
                        stroke_opacity=max(0, 1 - r/0.8)
                    ).move_to(electrons[i].get_center())
                    ripples.add(ripple)
            return ripples

        def get_bounded_ripples():
            rips = create_secondary_ripples()
            if len(rips) == 0:
                return VGroup()
            # Issue 39: Bounded secondary waves emission
            return self.place_in_area(rips, 'C1', 'E6', scale_factor=0.9)

        secondary_waves = always_redraw(get_bounded_ripples)
        
        self.add(secondary_waves)
        self.play(time_tracker.animate.set_value(8), run_time=4, rate_func=linear)

        # === Animation for Lecture Line 5 ===
        # Show the sum of original and secondary waves forming a slower resultant wave (#FFFFFF).
        self.lecture[4].set_color(WHITE)
        
        slower_wave = always_redraw(lambda: self.place_in_area(
            FunctionGraph(
                lambda x: 0.5 * np.sin(wave_freq * (x - (wave_speed * 0.6) * time_tracker.get_value())),
                x_range=[0, 5],
                color=WHITE,
                stroke_width=5
            ), 'B1', 'E6', scale_factor=1.0)
        )
        
        self.play(FadeOut(incoming_wave), FadeOut(secondary_waves))
        self.play(Create(slower_wave))
        self.play(time_tracker.animate.set_value(12), run_time=4, rate_func=linear)
        
        self.wait(2)
