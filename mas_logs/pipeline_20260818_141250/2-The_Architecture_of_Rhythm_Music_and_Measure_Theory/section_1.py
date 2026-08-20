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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisite: The Pulse of Time", [
            "Time flows in a constant steady pulse.",
            "We call each unit a beat.",
            "Imagine these as equally spaced dots."
        ])
        
        # Applying layout fixes requested
        self.place_in_area(self.lecture, 'B1', 'D2', scale_factor=0.8)

        # Assets
        heart = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/heart.svg")
        metronome = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/metronome.svg")
        drum = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/drum.svg")

        # Prep visuals
        dots_group = VGroup(*[Dot(color=WHITE) for _ in range(5)])
        line = Line(start=self.grid['C1'], end=self.grid['C6'], color=WHITE)
        grid_group = VGroup(line, dots_group)
        for i, dot in enumerate(dots_group):
            dot.move_to(self.grid[f'C{i+1}'])

        # Fix 1 & 2
        self.place_at_grid(dots_group, 'C2', scale_factor=1.0)
        self.place_in_area(grid_group, 'B2', 'E5', scale_factor=0.9)
        
        # === Animation for Lecture Line 1 ===
        # Display a steady pulse line across the center (#FFFFFF)
        self.play(FadeIn(heart.scale(0.5).next_to(self.lecture, DOWN)), FadeIn(line.set_color("#FFFFFF")))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        
        # === Animation for Lecture Line 2 ===
        # Highlight individual beats as growing circles (#FFFF00)
        self.play(FadeIn(metronome.scale(0.5).next_to(heart, RIGHT)))
        self.play(self.lecture[1].animate.set_color("#FFFF00"), dots_group.animate.set_color("#FFFF00").scale(1.5))
        
        # === Animation for Lecture Line 3 ===
        # Sequence beats in a linear progression (#00FF00)
        self.play(FadeIn(drum.scale(0.5).next_to(metronome, RIGHT)))
        self.play(self.lecture[2].animate.set_color("#00FF00"), dots_group.animate.set_color("#00FF00"))
        self.wait(1)
