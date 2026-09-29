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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Microscopic Mechanism: Forced Oscillations", [
            "Atoms behave like bound springs.",
            "Light waves force electron vibration.",
            "Induced dipoles create secondary waves."
        ])
        
        # Assets
        atom = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/atom.svg")
        spring = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/spring.svg")
        
        # === Animation for Lecture Line 1 ===
        # Atoms behave like bound springs.
        atom_spring = VGroup(atom, spring).arrange(RIGHT)
        self.place_at_grid(atom_spring, 'B3', scale_factor=1.2)
        self.play(FadeIn(atom_spring), self.lecture[0].animate.set_color(BLUE))

        # === Animation for Lecture Line 2 ===
        # Light waves force electron vibration.
        wave = FunctionGraph(lambda x: 0.2 * np.sin(3 * x), x_range=[-1, 1], color=YELLOW)
        self.place_at_grid(wave, 'C3', scale_factor=1.0)
        
        # updater for oscillation
        tracker = ValueTracker(0)
        atom_spring.add_updater(lambda m: m.set_x(self.grid['B3'][0] + 0.2 * np.sin(tracker.get_value())))
        
        self.play(Create(wave), self.lecture[1].animate.set_color(YELLOW))
        self.play(tracker.animate.set_value(4 * PI), run_time=3, rate_func=linear)
        atom_spring.remove_updater(atom_spring.updaters[0])
        
        # === Animation for Lecture Line 3 ===
        # Induced dipoles create secondary waves.
        secondary_wave = Circle(radius=0.2, color=GREEN, stroke_width=2)
        self.place_at_grid(secondary_wave, 'C3', scale_factor=0.8)
        self.play(
            FadeIn(secondary_wave),
            secondary_wave.animate.scale(3).set_stroke(opacity=0),
            self.lecture[2].animate.set_color(GREEN)
        )
        
        self.wait(1)
