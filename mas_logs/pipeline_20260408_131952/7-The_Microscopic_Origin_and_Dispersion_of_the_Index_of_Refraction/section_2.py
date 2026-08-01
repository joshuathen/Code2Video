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
        lines = [
            "Atoms contain electrons bound to nuclei like springs.",
            "Light’s electric field exerts a force on these electrons.",
            "The electrons then vibrate at the light's frequency."
        ]
        self.setup_layout("Prerequisite: Atoms as Oscillators", lines)

        # Assets
        atom_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/atom.svg"
        nucleus_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/nucleus.svg"

        # Tracker for oscillation
        time_tracker = ValueTracker(0)

        # === Animation for Lecture Line 1 ===
        # Colors: Grey (#AAAAAA), Red (#FF0000), Blue (#0000FF)
        self.lecture[0].set_color("#AAAAAA")
        
        # Single Atom Setup
        atom_bg = SVGMobject(atom_path).set_color(DARK_GREY).set_opacity(0.3)
        nucleus = SVGMobject(nucleus_path).set_color("#FF0000")
        electron = Dot(color="#0000FF", radius=0.15)
        
        # Positioning at center-right
        self.place_in_area(atom_bg, "B3", "E4", scale_factor=2.0)
        nucleus.move_to(atom_bg.get_center())
        nucleus.scale(0.3)
        
        # Initial position of electron above nucleus
        base_pos = nucleus.get_center()
        electron.move_to(base_pos + UP * 1.5)
        
        # Spring definition (8 segments)
        def get_spring_points(start, end):
            vec = end - start
            length = np.linalg.norm(vec)
            if length < 0.1: return [start, end]
            unit = vec / length
            perp = np.array([-unit[1], unit[0], 0]) * 0.2
            pts = [start]
            for i in range(1, 8):
                side = 1 if i % 2 == 1 else -1
                pts.append(start + (i/8)*vec + side*perp)
            pts.append(end)
            return pts

        spring = VMobject().set_stroke(color="#AAAAAA", width=3)
        spring.set_points_as_corners(get_spring_points(nucleus.get_center(), electron.get_center()))

        # Updaters for Line 1 oscillation
        def update_electron_line1(mob):
            # Slow natural oscillation
            offset = 1.5 + 0.2 * np.sin(time_tracker.get_value() * 3)
            mob.move_to(base_pos + UP * offset)

        def update_spring(mob):
            mob.set_points_as_corners(get_spring_points(nucleus.get_center(), electron.get_center()))

        self.play(FadeIn(atom_bg), FadeIn(nucleus), FadeIn(electron), Create(spring))
        
        electron.add_updater(update_electron_line1)
        spring.add_updater(update_spring)
        
        self.play(time_tracker.animate.set_value(2), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Color: Cyan (#00FFFF)
        self.lecture[1].set_color("#00FFFF")
        
        # Electric field arrow
        e_field_arrow = Arrow(UP*1.2, DOWN*1.2, color="#00FFFF", stroke_width=8)
        self.place_at_grid(e_field_arrow, "C1", scale_factor=0.8)
        
        e_field_label = Text("E-Field", font_size=18, color="#00FFFF").next_to(e_field_arrow, UP)
        
        # Switch electron updater to follow E-field
        electron.remove_updater(update_electron_line1)
        
        def update_electron_line2(mob):
            # Larger oscillation driven by E-field
            # Frequency matches time_tracker
            phase = time_tracker.get_value() * 5
            offset = 1.5 + 0.8 * np.sin(phase)
            mob.move_to(base_pos + UP * offset)
            
            # Sync arrow scaling
            e_field_arrow.set_y(self.grid["C1"][1] + 0.4 * np.sin(phase))
            e_field_arrow.scale_to_fit_height(1.2 + 0.8 * np.sin(phase))

        electron.add_updater(update_electron_line2)
        
        self.play(GrowArrow(e_field_arrow), Write(e_field_label))
        self.play(time_tracker.animate.set_value(5), run_time=3)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Color: White (#FFFFFF)
        self.lecture[2].set_color("#FFFFFF")
        
        # Transition to row of atoms
        self.play(
            FadeOut(atom_bg), FadeOut(nucleus), FadeOut(electron), FadeOut(spring),
            FadeOut(e_field_arrow), FadeOut(e_field_label)
        )
        
        electron.remove_updater(update_electron_line2)
        spring.remove_updater(update_spring)
        
        # Row of 5 atoms
        atoms = VGroup()
        nuclei = VGroup()
        electrons = VGroup()
        springs = VGroup()
        
        grid_cols = ["1", "2", "3", "4", "5", "6"]
        for col_idx in grid_cols:
            a = SVGMobject(atom_path).set_color(DARK_GREY).set_opacity(0.2).scale(0.4)
            self.place_at_grid(a, f"D{col_idx}")
            n = SVGMobject(nucleus_path).set_color("#FF0000").scale(0.1).move_to(a.get_center())
            e = Dot(color="#0000FF", radius=0.08).move_to(n.get_center() + UP * 0.6)
            s = VMobject().set_stroke(color="#AAAAAA", width=2)
            s.set_points_as_corners(get_spring_points(n.get_center(), e.get_center()))
            
            atoms.add(a)
            nuclei.add(n)
            electrons.add(e)
            springs.add(s)

        # Light wave
        wave = FunctionGraph(
            lambda x: 0.6 * np.sin(2 * PI * (x - 0.5) / 2.2),
            x_range=[0.5, 6.0],
            color=WHITE
        )
        wave.shift(RIGHT * 0.5 + DOWN * 0.2) # Centering roughly with row D

        def update_wave(mob):
            phase = time_tracker.get_value() * 4
            mob.become(
                FunctionGraph(
                    lambda x: 0.6 * np.sin(2 * PI * (x - 0.5) / 2.2 - phase),
                    x_range=[0.5, 6.0],
                    color=WHITE
                ).shift(DOWN * 0.8) # Row D height is approx -0.8
            )

        def update_row_electrons(mob_group):
            phase = time_tracker.get_value() * 4
            for i, e_dot in enumerate(mob_group):
                grid_x = self.grid[f"D{grid_cols[i]}"][0]
                # Match wave phase at that x
                val = 0.6 * np.sin(2 * PI * (grid_x - 0.5) / 2.2 - phase)
                base_y = self.grid[f"D{grid_cols[i]}"][1]
                e_dot.move_to([grid_x, base_y + 0.6 + val, 0])
                
                # Update corresponding spring
                springs[i].set_points_as_corners(
                    get_spring_points(nuclei[i].get_center(), e_dot.get_center())
                )

        self.play(FadeIn(atoms), FadeIn(nuclei), FadeIn(electrons), Create(springs))
        self.play(Create(wave))
        
        electrons.add_updater(update_row_electrons)
        wave.add_updater(update_wave)
        
        self.play(time_tracker.animate.set_value(10), run_time=5, rate_func=linear)
        self.wait(1)
