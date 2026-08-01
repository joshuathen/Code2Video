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
        # Initialize the layout with title and lecture lines
        self.setup_layout("Conclusion: The Power of 2^n - 1", [
            "- Binary counting perfectly models the Tower of Hanoi.",
            "- Total moves equal two to the power n minus one.",
            "- Each move is encoded in the binary sequence."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Initial highlight for the concluding statement
        self.play(self.lecture[0].animate.set_color("#FFFF00"), run_time=0.5)
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Display the formula 'M = 2^n - 1' in large cyan #00FFFF text,
        # accompanied by the tower icon [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/tower.svg].
        formula = MathTex("M = 2^n - 1", color="#00FFFF")
        tower_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tower.svg").set_color("#00FFFF")
        
        # Resolved Issue 44: Scale formula to 1.5
        self.place_in_area(formula, 'B2', 'B5', scale_factor=1.5)
        # Resolved Issue 32: Integrated Tower Asset
        self.place_in_area(tower_icon, 'A3', 'A4', scale_factor=0.8)
        
        self.play(
            self.lecture[1].animate.set_color("#00FFFF"),
            FadeIn(tower_icon),
            Write(formula)
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Demonstrate the magnitude for n = 64 disks.
        # Show n = 64 and the resulting massive number of moves below it.
        # Text '585 Billion Years' appears and slowly scales up in white #FFFFFF.
        
        n_display = MathTex("n = 64", color=WHITE)
        # Using Text for the long string of digits to ensure precision in layout
        moves_val = Text("18,446,744,073,709,551,615", color=WHITE)
        time_text = Text("585 Billion Years", color="#FFFFFF")

        self.place_at_grid(n_display, 'C3', scale_factor=1.0)
        # Resolved Issue 45: Scale moves_val to 0.7
        self.place_in_area(moves_val, 'D2', 'D5', scale_factor=0.7)
        # Resolved Issue 46: Place time_text in Row E
        self.place_in_area(time_text, 'E2', 'E5', scale_factor=1.0)

        self.play(
            self.lecture[2].animate.set_color("#FFFFFF"),
            FadeIn(n_display, shift=UP),
            FadeIn(moves_val, shift=UP)
        )
        self.wait(1.5)
        
        # Final dramatic reveal of the time scale
        self.play(
            FadeIn(time_text),
            run_time=1
        )
        self.play(
            time_text.animate.scale(1.3),
            run_time=3
        )
        self.wait(3)
