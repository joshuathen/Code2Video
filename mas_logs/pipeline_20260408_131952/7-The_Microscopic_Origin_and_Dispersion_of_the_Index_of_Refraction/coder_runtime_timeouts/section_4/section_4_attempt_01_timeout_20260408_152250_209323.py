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

class Section4Scene(TeachingScene):
    def construct(self):
        # Time tracker for animations
        time_tracker = ValueTracker(0)
        self.add(time_tracker)
        time_tracker.add_updater(lambda dt, dt_val=1/60: dt.increment_value(dt_val))

        lines = [
            "Electrons have a natural resonance frequency, often in UV.",
            "Pushing a swing works best at its natural rhythm.",
            "Frequencies closer to resonance cause stronger electron interactions.",
            "High-frequency blue light interacts more than low-frequency red.",
            "Stronger interaction leads to a higher refractive index."
        ]
        self.setup_layout("Color Dependence: The Resonance Factor", lines)

        # === Animation for Lecture Line 1 ===
        # Atom with vibrating electron (UV Resonance)
        self.play(self.lecture[0].animate.set_color(PURPLE_A))
        
        nucleus = Dot(color=WHITE)
        electron = Dot(color="#0000FF")
        spring = VMobject(color=GRAY)
        
        def update_spring(mob):
            t = time_tracker.get_value()
            y_offset = 0.4 * np.sin(15 * t)
            electron.move_to(nucleus.get_center() + UP * y_offset)
            points = [nucleus.get_center()]
            for i in range(1, 11):
                px = 0.05 * ((-1)**i)
                py = (y_offset / 10) * i
                points.append(nucleus.get_center() + np.array([px, py, 0]))
            mob.set_points_as_corners(points)

        atom_group = VGroup(nucleus, electron, spring)
        self.place_in_area(atom_group, "A1", "B2", scale_factor=1.0)
        
        res_label = Text("UV Resonance", font_size=18, color="#A020F0")
        self.place_at_grid(res_label, "B3", scale_factor=1.0)
        
        spring.add_updater(update_spring)
        self.play(Create(nucleus), Create(electron), Create(spring), FadeIn(res_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Swing asset at resonance
        self.play(self.lecture[1].animate.set_color(YELLOW))
        
        swing = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/swing.svg")
        swing.set_color(WHITE)
        self.place_in_area(swing, "A4", "B6", scale_factor=0.8)
        
        pivot = swing.get_top().copy()
        swing.save_state()
        
        def update_swing(mob):
            t = time_tracker.get_value()
            angle = 0.4 * np.sin(4 * t)
            mob.restore()
            mob.rotate(angle, about_point=pivot)
        
        swing.add_updater(update_swing)
        
        hand = Dot(color=GOLD_A).scale(2)
        def update_hand(m):
            m.move_to(swing.get_bottom() + RIGHT * 0.1)
        hand.add_updater(update_hand)

        self.play(DrawBorderThenFill(swing), FadeIn(hand))
        self.wait(2)
        # Note: updaters continue

        # === Animation for Lecture Line 3 ===
        # Side-by-side electron interaction (Red vs Blue)
        self.play(self.lecture[2].animate.set_color(BLUE))
        
        red_base = Dot(color=GRAY, radius=0.05)
        red_elec = Dot(color="#FF0000")
        self.place_at_grid(red_base, "C1")
        red_label = Text("Red (Low f)", font_size=16, color="#FF0000")
        self.place_at_grid(red_label, "D1")
        
        blue_base = Dot(color=GRAY, radius=0.05)
        blue_elec = Dot(color="#0000FF")
        self.place_at_grid(blue_base, "C4")
        blue_label = Text("Blue (High f)", font_size=16, color="#0000FF")
        self.place_at_grid(blue_label, "D4")

        red_elec.add_updater(lambda m: m.move_to(red_base.get_center() + UP * 0.2 * np.sin(3 * time_tracker.get_value())))
        blue_elec.add_updater(lambda m: m.move_to(blue_base.get_center() + UP * 0.6 * np.sin(8 * time_tracker.get_value())))
        
        self.play(FadeIn(red_base, red_elec, red_label, blue_base, blue_elec, blue_label))
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        # Waves passing through medium
        self.play(self.lecture[3].animate.set_color(BLUE_C))
        
        medium_line = Line(self.grid["E1"], self.grid["E6"], color=WHITE)
        medium_text = Text("Material Medium", font_size=14).next_to(medium_line, UP, buff=0.1)
        
        red_wave = VMobject(color="#FF0000")
        def update_red_wave(mob):
            t = time_tracker.get_value()
            points = [self.grid["E1"] + RIGHT * x * 0.6 + UP * 0.15 * np.sin(4*x - 6*t) for x in np.linspace(0, 8, 40)]
            mob.set_points_as_corners(points)
        
        blue_wave = VMobject(color="#0000FF")
        def update_blue_wave(mob):
            t = time_tracker.get_value()
            points = [self.grid["F1"] + RIGHT * x * 0.6 + UP * 0.3 * np.sin(10*x - 12*t) for x in np.linspace(0, 8, 40)]
            mob.set_points_as_corners(points)
            
        red_wave.add_updater(update_red_wave)
        blue_wave.add_updater(update_blue_wave)
        
        self.play(Create(medium_line), FadeIn(medium_text))
        self.play(Create(red_wave), Create(blue_wave))
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        # Summary Table
        self.play(self.lecture[4].animate.set_color(GOLD_E))
        
        blue_row = Text("Blue Light -> Strong Interaction -> Large Delay -> High n", font_size=16, color="#0000FF")
        red_row = Text("Red Light -> Weak Interaction -> Small Delay -> Low n", font_size=16, color="#FF0000")
        
        summary = VGroup(blue_row, red_row).arrange(DOWN, buff=0.3)
        self.place_in_area(summary, "A1", "F6", scale_factor=1.2)
        
        self.play(
            FadeOut(atom_group, res_label, swing, hand, red_base, red_elec, red_label, blue_base, blue_elec, blue_label, medium_line, medium_text, red_wave, blue_wave),
            FadeIn(summary)
        )
        self.wait(3)
