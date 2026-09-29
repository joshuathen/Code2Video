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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Synthesis: The Final Countdown", [
            "Identify the invariant.",
            "Test the extreme case.",
            "Connect abstractions to original constraints."
        ])

        # Assets
        clock = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/clock.svg")
        puzzle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/puzzle.svg")
        
        # Initial positions
        self.place_at_grid(clock, 'C4', scale_factor=0.8)
        self.place_at_grid(puzzle, 'C4', scale_factor=0.8)
        puzzle.set_opacity(0)
        
        # === Animation for Lecture Line 1 ===
        # Show clock
        self.play(FadeIn(clock), self.lecture[0].animate.set_color("#FFD700"), run_time=1)

        # === Animation for Lecture Line 2 ===
        # Rotate clock, switch to puzzle
        self.play(Rotate(clock, angle=2*PI), self.lecture[1].animate.set_color("#00CED1"), run_time=2)
        self.play(FadeOut(clock), FadeIn(puzzle), run_time=1)

        # === Animation for Lecture Line 3 ===
        # Flash
        self.play(self.lecture[2].animate.set_color("#00CED1"))
        flash = Flash(self.grid['C4'], color="#FFFFFF", line_length=0.5, num_lines=16)
        self.play(flash, run_time=1.5)
        self.wait(2)
