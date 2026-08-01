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
        time_tracker.add_updater(lambda dt, dt_val=1/30: dt.increment_value(dt_val))

        lines = [
            "Electrons have a natural resonance frequency, often in UV.",
            "Pushing a swing works best at its natural rhythm.",
            "Frequencies closer to resonance cause stronger electron interactions.",
            "High-frequency blue light interacts more than low-frequency red.",
            "Stronger interaction leads to a higher refractive index."
        ]
        self.setup_layout("Color Dependence: The Resonance Factor", lines)

        # Container for all dynamic stage objects for easy cleanup
        self.stage_mobjects = VGroup()

        # === Animation for Lecture Line 1 ===
        # Atom with vibrating electron (UV Resonance)
        self.play(self.lecture[0].animate.set_color("#A020F0"))
        
        nucleus = Dot(color=WHITE)
        electron = Dot(color="#0000FF")
        spring = VMobject(color=GRAY)
        
        def update_spring(mob):
            t = time_tracker.get_value()
            y_offset = 0.5 * np.sin(20 * t)
            electron.move_to(nucleus.get_center() + UP * y_offset)
            pts = []
            for i in range(11):
                py = (y_offset / 10) * i
                px = 0.05 * ((-1)**i) if 0 < i < 10 else 0
                pts.append(nucleus.get_center() + np.array([px, py, 0]))
            mob.set_points_as_corners(pts)

        atom_group = VGroup(nucleus, electron, spring)
        self.place_in_area(atom_group, "A1", "B2", scale_factor=0.8)
        
        res_label = Text("UV Resonance", font_size=18, color="#A020F0")
        self.place_at_grid(res_label, "B3", scale_factor=1.0)
        
        spring.add_updater(update_spring)
        self.stage_mobjects.add(atom_group, res_label)
        self.play(Create(nucleus), Create(electron), Create(spring), FadeIn(res_label))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Swing asset at resonance
        self.play(self.lecture[1].animate.set_color(YELLOW))
        
        swing = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/swing.svg")
        swing.set_color(WHITE)
        self.place_in_area(swing, "A4", "B6", scale_factor=0.8)
        
        pivot = swing.get_top()
        
        def update_swing(mob):
            t = time_tracker.get_value()
            # Resonance: High amplitude
            angle = 0.5 * np.sin(4 * t)
            mob.move_to(pivot)
            mob.shift(DOWN * mob.height/2)
            mob.rotate(angle, about_point=pivot)
        
        swing.add_updater(update_swing)
        
        hand = Dot(color=GOLD_A).scale(1.5)
        hand.add_updater(lambda m: m.move_to(swing.get_bottom() + RIGHT * 0.1))

        self.stage_mobjects.add(swing, hand)
        self.play(DrawBorderThenFill(swing), FadeIn(hand))
        self.wait(3)

        # === Animation for Lecture Line 3 ===
        # Side-by-side electron interaction (Red vs Blue)
        self.play(self.lecture[2].animate.set_color(BLUE))
        
        red_base = Dot(color=GRAY, radius=0.03)
        red_elec = Dot(color="#FF0000")
        self.place_at_grid(red_base, "C1")
        red_label = Text("Red: Small Response", font_size=14, color="#FF0000")
        self.place_at_grid(red_label, "D1")
        
        blue_base = Dot(color=GRAY, radius=0.03)
        blue_elec = Dot(color="#0000FF")
        self.place_at_grid(blue_base, "C4")
        blue_label = Text("Blue: Large Response", font_size=14, color="#0000FF")
        self.place_at_grid(blue_label, "D4")

        # Red moves little (far from resonance)
        red_elec.add_updater(lambda m: m.move_to(red_base.get_center() + UP * 0.15 * np.sin(3 * time_tracker.get_value())))
        # Blue moves much more (closer to UV resonance)
        blue_elec.add_updater(lambda m: m.move_to(blue_base.get_center() + UP * 0.5 * np.sin(10 * time_tracker.get_value())))
        
        self.stage_mobjects.add(red_base, red_elec, red_label, blue_base, blue_elec, blue_label)
        self.play(FadeIn(red_base, red_elec, red_label, blue_base, blue_elec, blue_label))
        self.wait(3)

        # === Animation for Lecture Line 4 ===
        # Waves passing through medium - ripples
        self.play(self.lecture[3].animate.set_color(BLUE_C))
        
        medium_box = Rectangle(width=5.5, height=1.5, color=WHITE, stroke_width=1)
        self.place_in_area(medium_box, "E1", "F6", scale_factor=1.0)
        
        red_ripple = VMobject(color="#FF0000")
        blue_ripple = VMobject(color="#0000FF")
        
        def update_red_wave(mob):
            t = time_tracker.get_value()
            pts = [self.grid["E1"] + RIGHT * x + UP * 0.1 * np.sin(3*x - 5*t) for x in np.linspace(0, 5, 20)]
            mob.set_points_as_corners(pts)
        
        def update_blue_wave(mob):
            t = time_tracker.get_value()
            # Higher frequency and higher amplitude ripples
            pts = [self.grid["F1"] + RIGHT * x + UP * 0.3 * np.sin(8*x - 12*t) for x in np.linspace(0, 5, 20)]
            mob.set_points_as_corners(pts)
            
        red_ripple.add_updater(update_red_wave)
        blue_ripple.add_updater(update_blue_wave)
        
        self.stage_mobjects.add(medium_box, red_ripple, blue_ripple)
        self.play(Create(medium_box), Create(red_ripple), Create(blue_ripple))
        self.wait(3)

        # === Animation for Lecture Line 5 ===
        # Summary Table
        self.play(self.lecture[4].animate.set_color(GOLD_E))
        
        blue_row = Text("Blue Light -> Strong Interaction -> Large Delay -> High n", font_size=16, color="#0000FF")
        red_row = Text("Red Light -> Weak Interaction -> Small Delay -> Low n", font_size=16, color="#FF0000")
        dispersion_text = Text("This is Dispersion.", font_size=20, color=WHITE)
        
        summary = VGroup(blue_row, red_row, dispersion_text).arrange(DOWN, buff=0.4)
        self.place_in_area(summary, "A1", "F6", scale_factor=1.1)
        
        self.play(
            FadeOut(self.stage_mobjects),
            FadeIn(summary)
        )
        self.wait(4)
