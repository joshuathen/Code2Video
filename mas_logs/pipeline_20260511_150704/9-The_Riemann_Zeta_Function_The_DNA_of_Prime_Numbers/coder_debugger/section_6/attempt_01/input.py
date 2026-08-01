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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup layout
        title = "Application: Cryptography and Prime Security"
        lines = [
            "Zeta's zeros act like the music of prime distribution.",
            "Understanding this music helps us predict prime locations.",
            "This security underpins modern digital encryption and privacy."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Primes sequence at the bottom
        primes_vals = ["2", "3", "5", "7", "11"]
        prime_mobs = VGroup(*[Text(val, font_size=36, color="#00FFFF") for val in primes_vals]).arrange(RIGHT, buff=0.5)
        # Fix for Issue 51: Aligned sequence within bottom row of right-side grid
        self.place_in_area(prime_mobs, 'F1', 'F6', scale_factor=0.8)
            
        self.play(FadeIn(prime_mobs))
        
        # Pulsing animation for primes
        self.play(
            *[Indicate(p, color="#00FFFF", scale_factor=1.4) for p in prime_mobs],
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(BLUE)
        
        # Zeta Zeros represented as dots
        zeta_zeros = VGroup()
        zero_spots = ["B1", "B3", "B5", "C2", "C4", "C6"]
        for spot in zero_spots:
            dot = Dot(color="#FF00FF")
            self.place_at_grid(dot, spot)
            zeta_zeros.add(dot)
            
        self.play(Create(zeta_zeros))
        
        # Visualizing "music" - small waves or pulses from zeros
        waves = VGroup()
        for zero in zeta_zeros:
            wave = Circle(radius=0.1, color="#FF00FF", stroke_opacity=0.5).move_to(zero.get_center())
            waves.add(wave)
            
        self.play(
            *[w.animate.scale(4).set_stroke(opacity=0) for w in waves],
            run_time=1.5
        )
        self.remove(waves)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(GREEN)
        
        # Silver padlock icon using SVG asset
        # Fix for Issue 34: Use [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/pad.svg]
        padlock = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/pad.svg").set_color("#C0C0C0")
        
        # Fix for Issue 49 and 50: Position at D3-E4 and scale to 0.6 to avoid clutter and overlap
        self.place_in_area(padlock, 'D3', 'E4', scale_factor=0.6)
        
        self.play(FadeIn(padlock))
        
        # Connections from Zeta Zeros to the Padlock
        connections = VGroup()
        for zero in zeta_zeros:
            line = Line(zero.get_center(), padlock.get_center(), color="#FFFFFF", stroke_width=1, stroke_opacity=0.6)
            connections.add(line)
            
        self.play(Create(connections), run_time=2)
        
        # Final Highlight: pulse the lock and connections
        self.play(
            Indicate(padlock, color="#C0C0C0"),
            connections.animate.set_stroke(width=2, opacity=1),
            run_time=1.5
        )
        self.wait(2)
