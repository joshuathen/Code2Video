from manim import *
import numpy as np

# Use the provided TeachingScene base class without modification.
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
        # Setup title and lecture lines
        title = "Summary and Real-World Impact"
        lines = [
            "- We compress these complex insights into a single core.",
            "- This math powers the security of our digital world.",
            "- Thus, the music of primes plays in perfect harmony."
        ]
        self.setup_layout(title, lines)
        
        # Color definitions for matching
        COLOR_CORE = YELLOW
        COLOR_SHIELD = BLUE_B
        COLOR_FINAL = "#90EE90"

        # === Animation for Lecture Line 1 ===
        # A montage of the prime grid, Euler gears, and the critical line shrinks into a central glowing core.
        
        # 1. Prime grid (montage component) - Fixed position as per Issue 47
        prime_grid = VGroup(*[Dot(radius=0.06, color=BLUE_A) for _ in range(9)])
        prime_grid.arrange_in_grid(rows=3, cols=3, buff=0.2)
        self.place_at_grid(prime_grid, 'C2', scale_factor=0.7)
        
        # 2. Euler gear (montage component) - Fixed position as per Issue 47
        gear_base = Circle(radius=0.4, color=WHITE, stroke_width=2)
        teeth = VGroup(*[
            Rectangle(width=0.1, height=0.1, color=WHITE, fill_opacity=1)
            .move_to([0.4 * np.cos(a), 0.4 * np.sin(a), 0])
            .rotate(a) 
            for a in np.linspace(0, 2*PI, 8, endpoint=False)
        ])
        gear = VGroup(gear_base, teeth)
        self.place_at_grid(gear, 'C5', scale_factor=0.8)
        
        # 3. Critical line (montage component)
        critical_line = Line(UP, DOWN, color=RED).scale(1.2)
        self.place_at_grid(critical_line, 'E3', scale_factor=0.8)
        
        montage = VGroup(prime_grid, gear, critical_line)
        
        # 4. Central Glowing Core (target of transformation) - Fixed position as per Issue 49
        core_center = Dot(radius=0.15, color=COLOR_CORE)
        core_halo = Circle(radius=0.3, color=COLOR_CORE, stroke_opacity=0.4).set_fill(COLOR_CORE, opacity=0.2)
        glowing_core = VGroup(core_center, core_halo)
        self.place_in_area(glowing_core, 'D3', 'E4', scale_factor=0.8)
        
        # Step 1: Animation
        self.lecture[0].set_color(COLOR_CORE)
        self.play(Create(montage), run_time=1.5)
        self.wait(0.5)
        self.play(ReplacementTransform(montage, glowing_core), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A digital shield icon [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/shield.svg]
        # labeled 'Encryption' appears, powered by energy pulses from the central core.
        
        # Step 2: Highlight line and create shield
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_SHIELD)
        
        # Using SVG Asset as per Issue 28
        shield_asset = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/shield.svg")
        shield_asset.set_color(COLOR_SHIELD).set_fill(COLOR_SHIELD, opacity=0.25)
        
        shield_label = Text("Encryption", font_size=16, color=WHITE)
        # Position label relative to shield asset
        shield_label.next_to(shield_asset, DOWN, buff=0.1)
        
        shield_group = VGroup(shield_asset, shield_label)
        # Fixed position and scale as per Issue 48
        self.place_in_area(shield_group, 'C2', 'F5', scale_factor=0.9)
        
        self.play(FadeIn(shield_group, scale=0.5))
        
        # energy pulses from central core
        for _ in range(3):
            pulse = Circle(radius=0.1, color=COLOR_CORE, stroke_width=3)
            pulse.move_to(glowing_core.get_center())
            self.add(pulse)
            self.play(
                pulse.animate.scale(12).set_stroke(opacity=0),
                run_time=0.9,
                rate_func=linear
            )
            self.remove(pulse)
            
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The screen fades to black with the final text 'The Music of Primes' appearing in soft green (#90EE90).
        
        # Step 3: Highlight line and final transition
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_FINAL)
        
        final_text = Text("The Music of Primes", font_size=42, color=COLOR_FINAL)
        
        self.play(
            FadeOut(self.lecture),
            FadeOut(self.title),
            FadeOut(shield_group),
            FadeOut(glowing_core),
            Write(final_text.move_to(ORIGIN)),
            run_time=2
        )
        self.wait(3)
