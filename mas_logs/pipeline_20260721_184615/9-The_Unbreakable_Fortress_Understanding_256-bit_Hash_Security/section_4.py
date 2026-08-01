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
        # Setup title and lecture lines
        title_str = "The Brute Force Challenge: Physics vs. Hacking"
        lecture_lines = [
            "Brute force means guessing every possible combination.",
            "Thermodynamics sets a physical limit on computing speed.",
            "Even covering Earth in supercomputers wouldn't be enough.",
            "The energy required exceeds the Sun's total output.",
            "Physics itself protects 256-bit encryption from being cracked."
        ]
        
        self.setup_layout(title_str, lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Spin a silver #C0C0C0 keyring [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/keyring.svg] against a white #FFFFFF lock icon.
        self.lecture[0].set_color("#C0C0C0")
        
        # Build Lock
        lock_body = Square(side_length=1.0, color="#FFFFFF", fill_opacity=0.5)
        lock_shackle = Arc(radius=0.4, start_angle=0, angle=PI, color="#FFFFFF")
        lock_shackle.shift(UP * 0.5)
        self.lock = VGroup(lock_body, lock_shackle)
        # Fix Issue 25: Adjust lock area to avoid overlap with energy limit text
        self.place_in_area(self.lock, "B3", "D4", scale_factor=0.8)
        
        # Build Keyring using Asset (Issue 18)
        self.keyring = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/keyring.svg")
        self.keyring.set_color("#C0C0C0")
        self.place_at_grid(self.keyring, "C2", scale_factor=0.6)
        
        self.play(FadeIn(self.lock), FadeIn(self.keyring))
        self.wait(1.5)
        # Spin keyring relative to the lock
        self.play(Rotate(self.keyring, angle=-2 * PI, about_point=self.lock.get_center(), run_time=2.5))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Thermodynamics sets a physical limit on computing speed.
        # Flash a red #FF0000 'Energy Limit' warning near a Sun icon.
        self.lecture[1].set_color("#FF0000")
        
        # Sun Icon
        sun_core = Circle(radius=0.4, color="#FFFF00", fill_opacity=1)
        rays = VGroup(*[
            Line(UP * 0.5, UP * 0.8, color="#FFFF00").rotate(a, about_point=ORIGIN)
            for a in np.linspace(0, 2 * PI, 8, endpoint=False)
        ])
        self.sun_icon = VGroup(sun_core, rays)
        self.place_at_grid(self.sun_icon, "B5", scale_factor=0.7)
        
        # Energy Limit Text
        self.energy_limit_txt = Text("ENERGY LIMIT", color="#FF0000", font_size=24)
        self.place_at_grid(self.energy_limit_txt, "C5", scale_factor=0.8)
        
        self.play(FadeOut(self.keyring), FadeIn(self.sun_icon))
        self.wait(1.5)
        # Use Indicate for highlighting
        self.play(Indicate(self.energy_limit_txt, color="#FF0000"))
        self.add(self.energy_limit_txt)
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # Even covering Earth in supercomputers wouldn't be enough.
        self.lecture[2].set_color("#00FF00")
        
        self.earth_circ = Circle(radius=0.5, color="#00FF00", fill_opacity=0.3)
        self.earth_label = Text("Earth", color="#00FF00", font_size=20)
        # Fix Issue 26: Reposition Earth and its label to avoid crowding
        self.place_at_grid(self.earth_circ, "E4", scale_factor=0.8)
        self.place_at_grid(self.earth_label, "F4", scale_factor=0.7)
        
        self.play(FadeIn(self.earth_circ), FadeIn(self.earth_label))
        self.wait(2.0)

        # === Animation for Lecture Line 4 ===
        # The energy required exceeds the Sun's total output.
        self.lecture[3].set_color("#FFFF00")
        self.wait(1.5)
        # Highlighting the sun
        self.play(Indicate(self.sun_icon, scale_factor=1.3, color="#FFFF00"))
        self.wait(1.5)

        # === Animation for Lecture Line 5 ===
        # Physics itself protects 256-bit encryption from being cracked.
        # Draw a long white #FFFFFF timeline exceeding the universe's age.
        self.lecture[4].set_color("#FFFFFF")
        
        # Timeline components
        timeline_line = Line(start=self.grid["E2"], end=self.grid["E6"], color="#FFFFFF")
        tick_now = Line(DOWN*0.1, UP*0.1, color="#FFFFFF").move_to(self.grid["E2"])
        label_now = Text("Now", color="#FFFFFF", font_size=16).next_to(tick_now, DOWN, buff=0.1)
        
        tick_univ = Line(DOWN*0.1, UP*0.1, color="#FFFFFF").move_to(self.grid["E2"] + RIGHT * 0.4)
        label_univ = Text("Universe Age", color="#FFFFFF", font_size=16).next_to(tick_univ, UP, buff=0.1)
        
        # Crack time arrow - extending way beyond the universe age
        crack_arrow = Arrow(start=self.grid["E2"], end=self.grid["E6"] + RIGHT * 2.0, color="#FF0000", buff=0)
        self.label_crack = Text("Time to Crack SHA-256", color="#FF0000", font_size=18)
        # Fix Issue 27: Use area positioning for long crack label
        self.place_in_area(self.label_crack, "F5", "F6", scale_factor=0.7)

        self.play(
            FadeOut(self.earth_circ), FadeOut(self.earth_label), 
            FadeOut(self.energy_limit_txt), FadeOut(self.sun_icon),
            FadeOut(self.lock)
        )
        self.play(Create(timeline_line), Create(tick_now), FadeIn(label_now))
        self.wait(1.0)
        self.play(Create(tick_univ), FadeIn(label_univ))
        self.wait(1.5)
        self.play(GrowArrow(crack_arrow), Write(self.label_crack))
        self.wait(2.0)
